from models import *
from utils import *

import os
import glob
import argparse
import sys
#import yaml
import math

import matplotlib.pyplot as plt

#import timm #only needed if downloading pretrained models
from datetime import datetime

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils2.renm_utils import * #load_weights_imagenet
import random

from einops import repeat
from einops.layers.torch import Rearrange

from vit_pytorch.vit import Transformer

def train(model, train_loader, loss_fn, device, input_dim, optimizer, epoch, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred = model(src=inputs, tgt=targets, tgt_mask=generate_subsequent_mask(model.latent_length).to(device))

        loss = loss_fn(pred, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss


if __name__ == "__main__":
    # loads in np train data/labels
    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    write_fpath = "/home/ahmadf/batch/sbatch.printSmallConv"

    write_to_file("Loaded in data.", filepath=write_fpath)

    # takes only the first L hemi
    train_data_np = train_data_np[:1, :, :, :]
    train_label_np = train_label_np[:1, :]

    # adds start token to train_labl_np
    #train_label_np = add_start_token_np(train_label_np)
    train_label_np = add_start_token_np(train_label_np, n=50) # specific for ProjectionConv

    # read numpy files into torch dataset and dataloader
    batch_size = 1
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"
    
    model = ProjectionConvFullTransformer(dim_model=96, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
                                          encoder_depth=3,
                                          nhead=6,
                                          encoder_mlp_dim=36, 
                                          decoder_input_dim=5000, 
                                          decoder_dim_feedforward=36,
                                          decoder_depth=6,
                                          dim_encoder_head=6,
                                          latent_length=50,
                                          dropout=0.1)
    
                            
    '''
    # SMALL120Dim
    model = FullTransformer(dim_model=120,
                            encoder_depth=3,
                            nhead=6,
                            encoder_mlp_dim=36, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=36,
                            decoder_depth=6,
                            dim_encoder_head=10,
                            latent_length=36,
                            dropout=0.1)
    '''
    '''
    # SMALL12Dim
    model = FullTransformer(dim_model=12,
                            encoder_depth=3,
                            nhead=6,
                            encoder_mlp_dim=36, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=36,
                            decoder_depth=6,
                            dim_encoder_head=2,
                            latent_length=36,
                            dropout=0.1)
    '''
    '''
    # SMALL100Seq
    model = FullTransformer(dim_model=36,
                            encoder_depth=3,
                            nhead=6,
                            encoder_mlp_dim=36, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=36,
                            decoder_depth=6,
                            dim_encoder_head=6,
                            latent_length=100,
                            dropout=0.1)
    '''
    '''
    # MASSIVE
    model = FullTransformer(dim_model=192,
                            encoder_depth=6,
                            nhead=6,
                            encoder_mlp_dim=192, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=192,
                            decoder_depth=6,
                            dim_encoder_head=32, # i fed this in as 32 on accident.. change this back if it starts complaining
                            latent_length=256, # maximum 512 or 256 -- but leads to GINORMOUS projection layers with >100million parameters 
                            dropout=0.1)
    '''
    '''
    # LARGE
    model = FullTransformer(dim_model=96,
                            encoder_depth=6,
                            nhead=6,
                            encoder_mlp_dim=192, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=192,
                            decoder_depth=6,
                            dim_encoder_head=32,
                            latent_length=64, # maximum 512 or 256 -- but leads to GINORMOUS projection layers with >100million parameters 
                            dropout=0.1)
    '''
    '''
    # SMALL
    model = FullTransformer(dim_model=36,
                            encoder_depth=3,
                            nhead=6,
                            encoder_mlp_dim=36, 
                            decoder_input_dim=4951, # this is built for an ICA dim _ --> schaefer dim 100 (4951 = n(n-1)/2 + 1 (the +1 is for the start token))
                            decoder_dim_feedforward=36,
                            decoder_depth=6,
                            dim_encoder_head=6,
                            latent_length=36,
                            dropout=0.1)
    '''    

    
    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, eps=1e-9) #orig lr=0.00001
    loss_fn = torch.nn.MSELoss()

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []
    forward_pass_mae_zeros_ith = []
    forward_pass_mae_zeros_TOith = []
    forward_pass_mae_ones_ith = []
    forward_pass_mae_ones_TOith = []
    forward_pass_mae_randn_ith = []
    forward_pass_mae_randn_TOith = []

    for epoch in range(1, 601):
        targets_, preds_, loss = train(model, train_loader, loss_fn, device, model.input_dim, optimizer, epoch)
        
        losses.append(float(loss.detach().numpy()))

        mae_epoch = np.mean(np.abs(targets_[0] - preds_[0]))
        maes.append(mae_epoch)

        write_to_file(f"##### EPOCH {epoch} #####", filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN MAEs:", filepath=write_fpath) 
        write_to_file(maes, filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN Losses:", filepath=write_fpath) 
        write_to_file(losses, filepath=write_fpath)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(train_loader):
                inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0)
                convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)
                write_to_file("MAE TRAIN Target vs TRAIN GENERATED:", filepath=write_fpath)
                write_to_file(np.mean(np.abs(targets.squeeze().numpy()[50:] - convDecoder_pred.squeeze().detach().numpy()[50:])), filepath=write_fpath)


        
        #torch.save(model.state_dict(), f"/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/SMALLConv_Model{epoch}.pt")

        #delete = f"/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/SMALLConv_Model{epoch-2}.pt"
        #if os.path.exists(delete):
        #    os.remove(delete)
        
        
        '''
        # Test forward pass
        if epoch % 50 == 0:
            write_to_file("###########################", filepath=write_fpath)
            model.eval()
            
            forward_targets = []
            forward_preds = []
            with torch.no_grad():
                for i, data in enumerate(train_loader):
                    inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0)
                    
                    predictions = greedy_decode(model=model, source=inputs, input_dim=model.input_dim, device=device, b=1)

                    forward_targets.append(targets.cpu().numpy())
                    forward_preds.append(predictions)

            forward_pass_mae_zeros_ith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][0].cpu().numpy())))
            forward_pass_mae_zeros_TOith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][1].cpu().numpy())))
            forward_pass_mae_ones_ith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][2].cpu().numpy())))
            forward_pass_mae_ones_TOith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][3].cpu().numpy())))
            forward_pass_mae_randn_ith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][4].cpu().numpy())))
            forward_pass_mae_randn_TOith.append(np.mean(np.abs(forward_targets[0] - forward_preds[0][5].cpu().numpy())))

            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_zeros_ith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_zeros_ith, filepath=write_fpath)
            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_zeros_TOith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_zeros_TOith, filepath=write_fpath)
            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_ones_ith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_ones_ith, filepath=write_fpath)
            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_ones_TOith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_ones_TOith, filepath=write_fpath)
            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_randn_ith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_randn_ith, filepath=write_fpath)
            write_to_file(f"EPOCH 1-{epoch} forward_pass_mae_randn_TOith MAEs:", filepath=write_fpath)
            write_to_file(forward_pass_mae_randn_TOith, filepath=write_fpath)

            write_to_file("###########################", filepath=write_fpath)
        '''
    

