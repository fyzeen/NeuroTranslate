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

from sklearn.decomposition import PCA

def read_regression_data(path):
    df = pd.read_csv(path)
    arr = df["coefficient"].to_numpy()
    reshaped_arr = arr.reshape((15, 320, 153))
    tensor = torch.from_numpy(reshaped_arr).float()
    return tensor.unsqueeze(0)

def read_regression_data_vanillakrakencoder(path):
    df = pd.read_csv(path)
    arr = df["coefficient"].to_numpy()
    arr = mesh_pca.transform(arr.reshape(1,-1))
    tensor = torch.from_numpy(arr).float()
    return tensor.unsqueeze(0)

def save_pred(pred, path, pad=None):
    if pad != None:
        pred = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy()[pad:], 0))
    else:
        pred = pca.inverse_transform(np.expand_dims(pred.squeeze().detach().numpy(), 0))
    np.save(path, pred)
    return    

    
if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    out_nodes = 100


    models = {
        
            "PCAConvTransformer_Large": {"model": ProjectionConvFullTransformer(dim_model=96, encoder_depth=6, nhead=6, encoder_mlp_dim=96, decoder_input_dim=264, decoder_dim_feedforward=96, decoder_depth=6, dim_encoder_head=16, latent_length=33, num_channels=15, num_patches=320, vertices_per_patch=153, dropout=0.1),
                                        "epoch": 90,
                                        "type": "encoder_decoder",
                                        "pad": 8}, 

            "PCAKrakLossEncoder_Large": {"model":SiT_nopool_linout(dim=384, depth=12, heads=6, mlp_dim=1536, num_patches=320, num_classes=256, num_channels=15, num_vertices=153, dim_head=64, dropout=0.1, emb_dropout=0.1),
                                     "epoch":350,
                                     "type":"SiT"},

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
                                          "type": "VAE_SiT"}, 

            "PCAVanillaKrakencoder": {"model": Krakencoder([256]), 
                                      "epoch": 90,
                                      "type": "krakencoder"}
                                        
    }

    # loads in actual train data to compute PCA
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")
    pca = PCA(n_components=256)
    pca.fit(train_label_np)

    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_mesh_flat = train_data_np.reshape(train_data_np.shape[0], -1)
    mesh_pca = PCA(n_components=256)
    mesh_pca.fit(train_mesh_flat)

    device = "cpu"
   
    # LOAD IN ALL MESHES (make into pytorch!)
    path = "/scratch/ahmadf/NeuroTranslate/SurfToNetmat/regressions/ICAd15_schfd100"
    # load L train regression mesh
    train_L_regression_mesh = read_regression_data(f"{path}/train_meshes_L_regression.csv")
    train_L_regression_mesh_krakencoder = read_regression_data_vanillakrakencoder(f"{path}/train_meshes_L_regression.csv")
    # load R train regression mesh
    train_R_regression_mesh = read_regression_data(f"{path}/train_meshes_R_regression.csv")
    train_R_regression_mesh_krakencoder = read_regression_data_vanillakrakencoder(f"{path}/train_meshes_R_regression.csv")
    # load L nontrain regression mesh
    nontrain_L_regression_mesh = read_regression_data(f"{path}/nontrain_meshes_L_regression.csv")
    nontrain_L_regression_mesh_krakencoder = read_regression_data_vanillakrakencoder(f"{path}/nontrain_meshes_L_regression.csv")
    # load R nontrain regression mesh
    nontrain_R_regression_mesh = read_regression_data(f"{path}/nontrain_meshes_R_regression.csv")
    nontrain_R_regression_mesh_krakencoder = read_regression_data_vanillakrakencoder(f"{path}/nontrain_meshes_R_regression.csv")

    save_path = "/scratch/ahmadf/NeuroTranslate/SurfToNetmat/regressions/ICAd15_schfd100/netmat_pred"

    for model_name in models:
        model = models[model_name]["model"]
        epoch = models[model_name]["epoch"]
        model_type = models[model_name]["type"]

        model.eval()
        model.to(device)
        
        with torch.no_grad():
            model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_name}_{epoch}.pt"))

            if model_type == "encoder_decoder":
                pad = models[model_name]["pad"]

                convDecoder_pred = greedy_decode_conv(model=model, source=train_L_regression_mesh, input_dim=model.input_dim, device=device, b=1)
                save_pred(convDecoder_pred, f"{save_path}/train_L_{model_name}.npy" , pad=pad)

                convDecoder_pred = greedy_decode_conv(model=model, source=train_R_regression_mesh, input_dim=model.input_dim, device=device, b=1) 
                save_pred(convDecoder_pred, f"{save_path}/train_R_{model_name}.npy" , pad=pad)

                convDecoder_pred = greedy_decode_conv(model=model, source=nontrain_L_regression_mesh, input_dim=model.input_dim, device=device, b=1) 
                save_pred(convDecoder_pred, f"{save_path}/nontrain_L_{model_name}.npy" , pad=pad)

                convDecoder_pred = greedy_decode_conv(model=model, source=nontrain_R_regression_mesh, input_dim=model.input_dim, device=device, b=1)
                save_pred(convDecoder_pred, f"{save_path}/nontrain_R_{model_name}.npy" , pad=pad)


            elif model_type == "SiT":
                pred, latent = model(img=train_L_regression_mesh)
                save_pred(pred, f"{save_path}/train_L_{model_name}.npy")  

                pred, latent = model(img=train_R_regression_mesh)
                save_pred(pred, f"{save_path}/train_R_{model_name}.npy") 

                pred, latent = model(img=nontrain_L_regression_mesh)
                save_pred(pred, f"{save_path}/nontrain_L_{model_name}.npy") 

                pred, latent = model(img=nontrain_R_regression_mesh)
                save_pred(pred, f"{save_path}/nontrain_R_{model_name}.npy") 
                

            elif model_type == "VAE_SiT":
                pred, mu, logvar = model(img=train_L_regression_mesh)
                save_pred(pred, f"{save_path}/train_L_{model_name}.npy")  

                pred, mu, logvar = model(img=train_R_regression_mesh)
                save_pred(pred, f"{save_path}/train_R_{model_name}.npy") 

                pred, lamu, logvartent = model(img=nontrain_L_regression_mesh)
                save_pred(pred, f"{save_path}/nontrain_L_{model_name}.npy") 

                pred, mu, logvar = model(img=nontrain_R_regression_mesh)
                save_pred(pred, f"{save_path}/nontrain_R_{model_name}.npy") 
                

            else:
                latent, pred = model(train_L_regression_mesh_krakencoder, 0, 0)
                save_pred(pred, f"{save_path}/train_L_{model_name}.npy")  

                latent, pred = model(train_R_regression_mesh_krakencoder, 0, 0)
                save_pred(pred, f"{save_path}/train_R_{model_name}.npy") 

                latent, pred = model(nontrain_L_regression_mesh_krakencoder, 0, 0)
                save_pred(pred, f"{save_path}/nontrain_L_{model_name}.npy") 

                latent, pred = model(nontrain_R_regression_mesh_krakencoder, 0, 0)
                save_pred(pred, f"{save_path}/nontrain_R_{model_name}.npy") 
                
    
    print("COMPLETE")

