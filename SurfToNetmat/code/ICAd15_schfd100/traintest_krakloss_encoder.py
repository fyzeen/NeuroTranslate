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

from krakloss import *


def train(model, train_loader, device, optimizer, epoch, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred, latent = model(img=inputs)
        
        # Output Losses
        Lr_corrI = correye(targets, pred) # corr mat of measured->predicted should be high along diagonal, loww off diagonal 
        Lr_mse = torch.FloatTensor(torch.nn.MSELoss()(pred, targets)) # MSE should be low
        Lr_marg = distance_loss(targets, pred, neighbor=True) # predicted X should be far from nearet ground truth X (for a different subject)

        # Latent Space Losses
        Lz_corrI = correye(latent, latent) # correlation matrix of latent space should be low off diagonal
        Lz_dist = distance_loss(latent, latent, neighbor=False) / (targets.size()[0]**2) # mean intersubject altent space distances should be high

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

def test(model, train_loader_fortesting, test_loader, device):
    model.eval()
    model.to(device)

    mse_train_list = []
    mae_train_list = []

    mse_test_list = []
    mae_test_list = []

    with torch.no_grad():
        for i, data in enumerate(train_loader_fortesting):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_train_list.append(mae)

            mse = np.mean( (targets.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_train_list.append(mse)

        for i, data in enumerate(test_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_test_list.append(mae)

            mse = np.mean( (targets.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_test_list.append(mse)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list)


if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    model_type = "KrakLossEncoder_Fisher"
    out_nodes = 100
    fisherZtransform = True


    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    write_fpath = f"/home/ahmadf/batch/sbatch.print{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    if fisherZtransform:
        train_label_np = np.arctanh(train_label_np)
        test_label_np = np.arctanh(test_label_np)

    # read numpy files into torch dataset and dataloader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # GraphTransformer
    model = SiT_nopool_linout(dim=384,
                              depth=12,
                              heads=6, 
                              mlp_dim=1536,
                              num_patches=320,
                              num_classes=4950,
                              num_channels=15,
                              num_vertices=153,
                              dim_head=64,
                              dropout=0.1,
                              emb_dropout=0.1)
    
    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, eps=1e-9)

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []

    train_mses_testing = []
    train_maes_testing = []
    test_mses = []
    test_maes = []

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
            train_mse, train_mae, test_mse, test_mae = test(model, train_loader_fortesting, test_loader, device)

            train_mses_testing.append(train_mse)
            train_maes_testing.append(train_mae)
            test_mses.append(test_mse)
            test_maes.append(test_mae)

            write_to_file(f"--- TESTING ---", filepath=write_fpath)
             
            write_to_file(f"--- train_mses_testing ---", filepath=write_fpath) 
            write_to_file(train_mses_testing, filepath=write_fpath) 
            write_to_file(f"--- train_maes_testing ---", filepath=write_fpath) 
            write_to_file(train_maes_testing, filepath=write_fpath)
            write_to_file(f"--- test_mses ---", filepath=write_fpath) 
            write_to_file(test_mses, filepath=write_fpath)
            write_to_file(f"--- test_maes ---", filepath=write_fpath) 
            write_to_file(test_maes, filepath=write_fpath)

        else:
            delete = f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_{epoch}.pt"
            if os.path.exists(delete):
                os.remove(delete)
        
    

