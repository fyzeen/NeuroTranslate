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


def encoder_decoder_test(model, train_loader_fortesting, test_loader, device, pca, pad):
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

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - convDecoder_pred.squeeze().detach().numpy()[pad:]))
            mae_train_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - convDecoder_pred.squeeze().detach().numpy()[pad:])**2 )
            mse_train_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(convDecoder_pred.squeeze().detach().numpy()[pad:], 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=1)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - convDecoder_pred.squeeze().detach().numpy()[pad:]))
            mae_test_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - convDecoder_pred.squeeze().detach().numpy()[pad:])**2 )
            mse_test_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(convDecoder_pred.squeeze().detach().numpy()[pad:], 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_test_list.append(corr)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list), np.mean(corr_train_list), np.mean(corr_test_list), corr_train_list, corr_test_list

def encoder_test(model, train_loader_fortesting, test_loader, device, pca, model_name):
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

            if "VAE" in model_name or "Variational" in model_name:
                pred, mu, log_var = model(img=inputs)
            else:
                pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_train_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_train_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_train_list.append(corr)
        
        for i, data in enumerate(test_loader):
            inputs, targets_transform, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            if "VAE" in model_name or "Variational" in model_name:
                pred, mu, log_var = model(img=inputs)
            else:
                pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_test_list.append(mae)

            mse = np.mean( (targets_transform.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_test_list.append(mse)

            inverse_pca = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
            corr = np.corrcoef(targets.squeeze().numpy(), inverse_pca.squeeze(0))[0,1]
            corr_test_list.append(corr)
    
    return np.mean(mse_train_list), np.mean(mae_train_list), np.mean(mse_test_list), np.mean(mae_test_list), np.mean(corr_train_list), np.mean(corr_test_list), corr_train_list, corr_test_list

if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    out_nodes = 100

    # General form of models dictionary:
    #    model_types["name of model"] will return another dictionary with {"model": model object, "epoch": epoch_of_highest_performing_model, "type":"encoder_decoder" or "encoder_only", "pad": padding (int or none)}


    models = {
        "PCAConvTransformer_Shallow": {"model": ProjectionConvFullTransformer(dim_model=96, encoder_depth=2, nhead=6, encoder_mlp_dim=96, decoder_input_dim=264, decoder_dim_feedforward=96, decoder_depth=2, dim_encoder_head=16, latent_length=33, num_channels=15, num_patches=320, vertices_per_patch=153, dropout=0.1),
                                     "epoch": 190,
                                     "type": "encoder_decoder",
                                     "pad": 8},
        "PCAConvTransformer_Large": {"model": ProjectionConvFullTransformer(dim_model=96, encoder_depth=6, nhead=6, encoder_mlp_dim=96, decoder_input_dim=264, decoder_dim_feedforward=96, decoder_depth=6, dim_encoder_head=16, latent_length=33, num_channels=15, num_patches=320, vertices_per_patch=153, dropout=0.1),
                                     "epoch": 90,
                                     "type": "encoder_decoder",
                                     "pad": 8},  
        "PCAConvTransformer_SmallDim": {"model": ProjectionConvFullTransformer(dim_model=24, encoder_depth=6, nhead=4, encoder_mlp_dim=24, decoder_input_dim=264, decoder_dim_feedforward=24, decoder_depth=6, dim_encoder_head=6, latent_length=33, num_channels=15, num_patches=320, vertices_per_patch=153, dropout=0.1),
                                     "epoch": 500,
                                     "type": "encoder_decoder",
                                     "pad": 8},
        "PCAConvTransformer_Tiny": {"model": ProjectionConvFullTransformer(dim_model=24, encoder_depth=2, nhead=4, encoder_mlp_dim=24, decoder_input_dim=264, decoder_dim_feedforward=24, decoder_depth=2, dim_encoder_head=6, latent_length=33, num_channels=15, num_patches=320, vertices_per_patch=153, dropout=0.1),
                                     "epoch": 500,
                                     "type": "encoder_decoder",
                                     "pad": 8},

        "PCAKrakLossEncoder_Shallow": {"model": SiT_nopool_linout(dim=384,
                              depth=2,
                              heads=6, 
                              mlp_dim=1536,
                              num_patches=320,
                              num_classes=256,
                              num_channels=15,
                              num_vertices=153,
                              dim_head=64,
                              dropout=0.1,
                              emb_dropout=0.1),
                                     "epoch": 130,
                                     "type": "encoder_only"},
        "PCAKrakLossEncoder_SmallDim": {"model": SiT_nopool_linout(dim=48,
                              depth=12,
                              heads=6, 
                              mlp_dim=96,
                              num_patches=320,
                              num_classes=256,
                              num_channels=15,
                              num_vertices=153,
                              dim_head=8,
                              dropout=0.1,
                              emb_dropout=0.1),
                                     "epoch": 240,
                                     "type": "encoder_only"},
        "PCAKrakLossEncoder_Tiny": {"model": SiT_nopool_linout(dim=48,
                              depth=2,
                              heads=6, 
                              mlp_dim=96,
                              num_patches=320,
                              num_classes=256,
                              num_channels=15,
                              num_vertices=153,
                              dim_head=8,
                              dropout=0.1,
                              emb_dropout=0.1),
                                     "epoch": 510,
                                     "type": "encoder_only"},
        "PCAKrakLossEncoder_Large": {"model":SiT_nopool_linout(dim=384, depth=12, heads=6, mlp_dim=1536, num_patches=320, num_classes=256, num_channels=15, num_vertices=153, dim_head=64, dropout=0.1, emb_dropout=0.1),
                                     "epoch":350,
                                     "type":"encoder_only"},
        
        "PCAVAEKrakLossEncoder_Shallow": {"model":VariationalSiT_nopool_linout(dim=384,
                                         depth=2,
                                         heads=6, 
                                         mlp_dim=1536, # was originally 1536
                                         VAE_latent_dim=256,
                                         num_patches=320,
                                         num_classes=256,
                                         num_channels=15,
                                         num_vertices=153,
                                         dim_head=64,
                                         dropout=0.1,
                                         emb_dropout=0.1),
                                          "epoch":510,
                                          "type": "encoder_only"}, 
                                          
        "PCAVAEKrakLossEncoder_SmallDim": {"model":VariationalSiT_nopool_linout(dim=48,
                                         depth=12,
                                         heads=6, 
                                         mlp_dim=48, 
                                         VAE_latent_dim=48,
                                         num_patches=320,
                                         num_classes=256,
                                         num_channels=15,
                                         num_vertices=153,
                                         dim_head=8,
                                         dropout=0.1,
                                         emb_dropout=0.1),
                                          "epoch":80,
                                          "type": "encoder_only"},
        "PCAVAEKrakLossEncoder_Tiny": {"model": VariationalSiT_nopool_linout(dim=48,
                                         depth=2,
                                         heads=6, 
                                         mlp_dim=48, 
                                         VAE_latent_dim=48,
                                         num_patches=320,
                                         num_classes=256,
                                         num_channels=15,
                                         num_vertices=153,
                                         dim_head=8,
                                         dropout=0.1,
                                         emb_dropout=0.1),
                                          "epoch":580,
                                          "type": "encoder_only"}
                                                          
    }


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


    test_batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

    train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
    train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)


    write_fpath = f"/home/ahmadf/batch/sbatch.printNonDemeaned_Corrs_AllModels_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    device = "cpu"

    for model_name in models:
        model = models[model_name]["model"]
        epoch = models[model_name]["epoch"]
        model_type = models[model_name]["type"]

        if model_type == "encoder_decoder":
            pad = models[model_name]["pad"]
            temp_train_transform = add_start_token_np(train_transform, n=pad)
            temp_test_transform = add_start_token_np(test_transform, n=pad)


            temp_test_batch_size = 1
            temp_test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_transform).float(), torch.from_numpy(test_label_np).float())
            temp_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = test_batch_size, shuffle=False, num_workers=10)

            temp_train_dataset_fortesting = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_transform).float(), torch.from_numpy(train_label_np).float())
            temp_train_loader_fortesting = torch.utils.data.DataLoader(train_dataset_fortesting, batch_size = test_batch_size, shuffle=False, num_workers=10)
            

            model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_name}_{epoch}.pt"))
            train_mse, train_mae, test_mse, test_mae, train_corr, test_corr, train_corr_list, test_corr_list = encoder_decoder_test(model, temp_train_loader_fortesting, temp_test_loader, device, pca, pad)

        else:
            model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_name}_{epoch}.pt"))
            train_mse, train_mae, test_mse, test_mae, train_corr, test_corr, train_corr_list, test_corr_list = encoder_test(model, train_loader_fortesting, test_loader, device, pca, model_name=model_name)
        
        write_to_file(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", filepath=write_fpath) 
        write_to_file(f"--------------------------------", filepath=write_fpath) 
        write_to_file(f"      --- {model_name} ---      ", filepath=write_fpath) 
        write_to_file(f"--------------------------------", filepath=write_fpath) 
        write_to_file(f"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", filepath=write_fpath) 
        write_to_file(f"--- test_corrs (NO DEMEANED) ---", filepath=write_fpath) 
        write_to_file(test_corr, filepath=write_fpath)
        write_to_file(test_corr_list, filepath=write_fpath)
        write_to_file(f"--- train_corrs (NO DEMEANED) ---", filepath=write_fpath) 
        write_to_file(train_corr, filepath=write_fpath)
        write_to_file(train_corr_list, filepath=write_fpath)




