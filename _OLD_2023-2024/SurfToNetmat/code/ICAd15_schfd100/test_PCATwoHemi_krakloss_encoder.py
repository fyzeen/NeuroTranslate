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
            hemi1, hemi2, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze(), data[3].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(hemi1, hemi2)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_train_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_train_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            hemi1, hemi2, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze(), data[3].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(hemi1, hemi2)

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
    model_type = "PCAKrakLossEncoder_TwoHemi_Large"
    out_nodes = 100


    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    # updating shape of train/test data/lables
    num_train_subj = int(train_data_np.shape[0] / 2)
    num_test_subj = int(test_data_np.shape[0] / 2)

    train_label_np = train_label_np[:num_train_subj]
    test_label_np = test_label_np[:num_test_subj]

    _, c, p, v = train_data_np.shape
    train_data_np_hemi1 = np.zeros([num_train_subj, c, p, v])
    train_data_np_hemi2 = np.zeros([num_train_subj, c, p, v])
    test_data_np_hemi1 = np.zeros([num_test_subj, c, p, v])
    test_data_np_hemi2 = np.zeros([num_test_subj, c, p, v])

    for i in range(num_train_subj):
        train_data_np_hemi1[i] = train_data_np[i]
        train_data_np_hemi2[i] = train_data_np[i+num_train_subj]
    
    for i in range(num_test_subj):
        test_data_np_hemi1[i] = test_data_np[i]
        test_data_np_hemi2[i] = test_data_np[i+num_test_subj]

    # compute pca on train
    pca = PCA(n_components=256)
    pca.fit(train_label_np)
    train_transform = pca.transform(train_label_np)
    test_transform = pca.transform(test_label_np)

    # read numpy files into torch dataset and dataloader
    batch_size = 16
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np_hemi1).float(), torch.from_numpy(train_data_np_hemi2).float(), torch.from_numpy(train_transform).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np_hemi1).float(), torch.from_numpy(test_data_np_hemi2).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np_hemi1).float(), torch.from_numpy(train_data_np_hemi2).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)

    write_fpath = f"/home/ahmadf/batch/temp1/sbatch.printTEST{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # Encoder_Large
    model = TwoHemi_SiT_nopool_linout(dim=384,
                              depth=12,
                              heads=6, 
                              mlp_dim=1536,
                              latent_dim = 128,
                              num_patches=320,
                              num_classes=256,
                              num_channels=15,
                              num_vertices=153,
                              dim_head=64,
                              dropout=0.1,
                              emb_dropout=0.1)
    
    model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_390.pt"))
    

    mean_train_label = np.mean(train_label_np, axis=0)

    maes = []
    losses = []

    train_mses_testing = []
    train_maes_testing = []
    train_corrs = []
    test_mses = []
    test_maes = []
    test_corrs = []

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