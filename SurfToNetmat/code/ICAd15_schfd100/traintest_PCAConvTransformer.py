import sys
import os
import math

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

from utils.models import *
from utils.utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA


def train(model, train_loader, loss_fn, device, input_dim, optimizer, epoch, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze() #.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred, _ = model(src=inputs, tgt=targets, tgt_mask=generate_subsequent_mask(model.latent_length).to(device))

        loss = loss_fn(pred, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss

def test(model, train_loader_fortesting, test_loader, device, pca):
    model.eval()
    model.to(device)

    mse_train_list = []
    mae_train_list = []
    corr_train_list = []

    mse_test_list = []
    mae_test_list = []
    corr_test_list = []

    with torch.no_grad():
        for i, data in enumerate(train_loader_fortesting):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=1)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:]))
            mae_train_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])**2 )
            mse_train_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(convDecoder_pred.squeeze().detach().numpy()[pad:], 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0, 1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=1)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:]))
            mae_test_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])**2 )
            mse_test_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(convDecoder_pred.squeeze().detach().numpy()[pad:], 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0, 1]
            corr_test_list.append(corr)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list), np.mean(corr_train_list), np.mean(corr_test_list)


if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    model_type = "PCAConvTransformer"
    out_nodes = 100

    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    # compute pca on train
    pca = PCA(n_components=1000)
    pca.fit(train_label_np)
    train_transform = pca.transform(train_label_np)
    test_transform = pca.transform(test_label_np)

    # adds start token to *_label_np
    pad=10
    train_transform = add_start_token_np(train_transform, n=pad)
    test_transform = add_start_token_np(test_transform, n=pad)

    # read numpy files into torch dataset and dataloader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_transform).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)

    write_fpath = f"/home/ahmadf/batch/temp/sbatch.print{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # ConvTransformer
    model = ProjectionConvFullTransformer(dim_model=96, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
                                          encoder_depth=6,
                                          nhead=6,
                                          encoder_mlp_dim=36, 
                                          decoder_input_dim=1010, 
                                          decoder_dim_feedforward=36,
                                          decoder_depth=6,
                                          dim_encoder_head=16,
                                          latent_length=101,
                                          num_channels=15,
                                          num_patches=320, 
                                          vertices_per_patch=153,
                                          dropout=0.1)


    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)
    loss_fn = torch.nn.MSELoss()

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []

    train_mses_testing = []
    train_maes_testing = []
    train_corrs = []
    test_mses = []
    test_maes = []
    test_corrs = []

    for epoch in range(0, 601):
        targets_, preds_, loss = train(model, train_loader, loss_fn, device, model.input_dim, optimizer, epoch)
        
        losses.append(float(loss.detach().numpy()))

        mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))
        maes.append(mae_epoch)

        write_to_file(f"##### EPOCH {epoch} #####", filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN MAEs:", filepath=write_fpath) 
        write_to_file(maes, filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN Losses:", filepath=write_fpath) 
        write_to_file(losses, filepath=write_fpath)
        
        torch.save(model.state_dict(), f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_{epoch}.pt")

        if epoch%10 == 0:
            train_mse, train_mae, test_mse, test_mae, train_corr, test_corr = test(model, train_loader_fortesting, test_loader, device, pca)

            train_mses_testing.append(train_mse)
            train_maes_testing.append(train_mae)
            train_corrs.append(train_corr)
            test_mses.append(test_mse)
            test_maes.append(test_mae)
            test_corrs.append(test_corr)

            write_to_file(f"--- TESTING ---", filepath=write_fpath)
             
            write_to_file(f"--- train_mses_testing ---", filepath=write_fpath) 
            write_to_file(train_mses_testing, filepath=write_fpath) 
            write_to_file(f"--- train_maes_testing ---", filepath=write_fpath) 
            write_to_file(train_maes_testing, filepath=write_fpath)
            write_to_file(f"--- train_corrs_testing ---", filepath=write_fpath) 
            write_to_file(train_corrs, filepath=write_fpath)
            write_to_file(f"--- test_mses ---", filepath=write_fpath) 
            write_to_file(test_mses, filepath=write_fpath)
            write_to_file(f"--- test_maes ---", filepath=write_fpath) 
            write_to_file(test_maes, filepath=write_fpath)
            write_to_file(f"--- test_corrs ---", filepath=write_fpath) 
            write_to_file(test_corrs, filepath=write_fpath)

        else:
            delete = f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_{epoch}.pt"
            if os.path.exists(delete):
                os.remove(delete)

        
    

