import sys
import os
import math

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

from utils.models import *
from utils.utils import *
from utils.krakencoder_model import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from krakloss import *

from sklearn.decomposition import PCA



def train(model, train_loader, device, optimizer, epoch, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        latent, pred = model(inputs, 0, 0)
        
        # Output Losses
        Lr_corrI = correye(targets, pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(pred, targets)) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr = Lr_corrI + Lr_marg + (1000 * Lr_mse) 
        Lz = Lz_corrI + Lz_dist

        loss = Lr + (10 * Lz) # weighting Lz with 10 (from Krakencoder)

        mae = torch.nn.L1Loss()(pred, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss, mae


def test(model, train_loader_fortesting, test_loader, device, pca, mean_train_label):
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

            latent, pred = model(inputs, 0, 0)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_train_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_train_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy() - mean_train_label, inverse_pca.squeeze(0) - mean_train_label)[0,1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            latent, pred = model(inputs, 0, 0)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_test_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_test_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy() - mean_train_label, inverse_pca.squeeze(0) - mean_train_label)[0,1]
            corr_test_list.append(corr)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list), np.mean(corr_train_list), np.mean(corr_test_list)


if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    model_type = "PCAVanillaKrakencoder_Deep"
    out_nodes = 100


    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    # compute pca on train
    pca = PCA(n_components=256)
    pca.fit(train_label_np)
    train_transform = pca.transform(train_label_np)
    test_transform = pca.transform(test_label_np)

    # mesh PCAs
    train_mesh_flat = train_data_np.reshape(train_data_np.shape[0], -1) 
    test_mesh_flat = test_data_np.reshape(test_data_np.shape[0], -1) 
    
    mesh_pca = PCA(n_components=256)
    mesh_pca.fit(train_mesh_flat)
    train_data_transform = mesh_pca.transform(train_mesh_flat)
    test_data_transform = mesh_pca.transform(test_mesh_flat)



    # read numpy files into torch dataset and dataloader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_transform).float(), torch.from_numpy(train_transform).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_transform).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_transform).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)

    write_fpath = f"/home/ahmadf/batch/temp/sbatch.print{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    # Vanilla Krakencoder
    #model = Krakencoder([256])

    # OneHidden512
    #model = Krakencoder([256], hiddenlayers=[512])

    # Latent512
    model = Krakencoder([256], latentsize=512)

    # Deep
    #model = Krakencoder([256], hiddenlayers=[128, 128, 128])
    
    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)

    mean_train_label = np.mean(train_label_np, axis=0)

    maes = []
    losses = []

    train_mses_testing = []
    train_maes_testing = []
    train_corrs = []
    test_mses = []
    test_maes = []
    test_corrs = []

    for epoch in range(0, 601):
        targets_, preds_, loss, mae = train(model, train_loader, device, optimizer, epoch)
        
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
            train_mse, train_mae, test_mse, test_mae, train_corr, test_corr = test(model, train_loader_fortesting, test_loader, device, pca, mean_train_label)

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