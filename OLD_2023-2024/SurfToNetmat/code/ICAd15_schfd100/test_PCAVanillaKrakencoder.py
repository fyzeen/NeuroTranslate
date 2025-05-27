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
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            latent, pred = model(inputs, 0, 0)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_test_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_test_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_test_list.append(corr)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list), np.mean(corr_train_list), np.mean(corr_test_list)


if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    out_nodes = 100


    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/1L_train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/1L_train_labels.npy")

    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/1L_test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/1L_test_labels.npy")

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

    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_transform).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_transform).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)

    write_fpath = f"/home/ahmadf/batch/temp2/sbatch.printTestVanillaKrakencoders_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    
    models = {"PCAVanillaKrakencoder": [Krakencoder([256]), 90]}

    mean_train_label = np.mean(train_label_np, axis=0)

    for model_name in models:
        model = models[model_name][0]
        epoch = models[model_name][1]
        model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_name}_{epoch}.pt"))

  
        train_mse, train_mae, test_mse, test_mae, train_corr, test_corr = test(model, train_loader_fortesting, test_loader, device, pca, mean_train_label)

        write_to_file(f"--- TESTING ---", filepath=write_fpath)   
        write_to_file(f"--- train_corrs_testing ---", filepath=write_fpath) 
        write_to_file(train_corr, filepath=write_fpath)
        write_to_file(f"--- test_corrs ---", filepath=write_fpath) 
        write_to_file(test_corr, filepath=write_fpath)