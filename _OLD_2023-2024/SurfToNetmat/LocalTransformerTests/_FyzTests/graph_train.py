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
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred = model(src=inputs, tgt=targets, tgt_mask=generate_subsequent_mask(model.latent_length).to(device))

        lower_triangle_mask = torch.tril(torch.ones(model.latent_length, model.latent_length), diagonal=1).bool()
        lower_triangle_mask = lower_triangle_mask.unsqueeze(0).expand(pred.shape[0], model.latent_length, model.latent_length)
        loss = loss_fn(pred.masked_select(lower_triangle_mask), targets.masked_select(lower_triangle_mask))

        #loss = loss_fn(pred, targets)

        loss2func = torch.nn.L1Loss()
        loss2 = loss2func(pred.masked_select(lower_triangle_mask), targets.masked_select(lower_triangle_mask))

        #loss2 = loss2func(pred, targets)


        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss, loss2


if __name__ == "__main__":
    # loads in np train data/labels
    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    # make netmat and add start node(s) -- you need to have an EVEN number of NODES so that model_dim can be even
    train_label_np = make_nemat_allsubj(train_label_np, 100)
    train_label_np = add_start_node(train_label_np)
    train_label_np = add_start_node(train_label_np)

    # read numpy files into torch dataset and dataloader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    # write to file
    write_fpath = "/home/ahmadf/batch/sbatch.printLowerTriLossGraph_Model"
    write_to_file("Loaded in data.", filepath=write_fpath)


    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # GraphTransformer
    model = GraphTransformer(dim_model=102, 
                             encoder_depth=6, 
                             nhead=6, 
                             encoder_mlp_dim=102,
                             decoder_input_dim=102, 
                             decoder_dim_feedforward=102,
                             decoder_depth=6,
                             dim_encoder_head=17, 
                             latent_length=102,
                             dropout=0.1)
    
    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.000001, eps=1e-9)
    loss_fn = torch.nn.MSELoss()

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []

    for epoch in range(1, 601):
        targets_, preds_, loss, loss2 = train(model, train_loader, loss_fn, device, model.input_dim, optimizer, epoch)
        
        losses.append(float(loss.detach().numpy()))

        #mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))
        mae_epoch = float(loss2.detach().numpy())
        maes.append(mae_epoch)

        write_to_file(f"##### EPOCH {epoch} #####", filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN MAEs:", filepath=write_fpath) 
        write_to_file(maes, filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN Losses:", filepath=write_fpath) 
        write_to_file(losses, filepath=write_fpath)
        
        #torch.save(model.state_dict(), f"/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/LowerTriLossGraphModel_{epoch}.pt")

        #delete = f"/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/LowerTriLossGraphModel_{epoch-2}.pt"
        #if os.path.exists(delete):
        #    os.remove(delete)

        
    

