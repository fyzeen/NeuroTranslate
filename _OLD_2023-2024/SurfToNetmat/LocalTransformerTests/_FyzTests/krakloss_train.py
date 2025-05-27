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


def xycorr(x,y,axis=1):
    """
    **FROM KRAKENCODER.loss.py**

    Compute correlation between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    NOTE: in train.py we always call cc=xycorr(Ctrue, Cpredicted)
    which means cc[i,:] is cc[true subject i, predicted for all subjects]
    and thus top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if torch.is_tensor(x):
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
        cc=torch.matmul(cx,cy.t())
    else:
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
        cc=np.matmul(cx,cy.T)
    return cc


def correye(x,y):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: mean squared error between pairwise correlation matrix for xycorr(x,y) and identity matrix
    (i.e., want diagonal to be near 1, off-diagonal to be near 0)
    """
    cc=xycorr(x,y)
    #need keepdim for some reason now that correye and enceye are separated
    loss=torch.norm(cc-torch.eye(cc.shape[0],device=cc.device),keepdim=True)
    return loss

def distance_loss(x,y, margin=None, neighbor=False):
    """
    **FROM KRAKENCODER.loss.py**

    Loss function: difference between self-distance and other-distance for x and y, with optional margin
    If neighbor=True, reconstruction loss applies only to nearest neighbor distance, otherwise to mean distance between all
        off-diagonal pairs.
    
    Inputs:
    x: torch tensor (Nsubj x M), generally the measured data
    y: torch tensor (Nsubj x M), generally the predicted data
    margin: float, optional margin for distance loss (distance above margin is penalized, below is ignored)
    neighbor: bool, (optional, default=False), True for maximizing nearest neighbor distance, False for maximizing mean distance
    
    Returns: 
    loss: torch FloatTensor, difference between self-distance and other-distance
    """
    
    d=torch.cdist(x,y)
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance -- avg distance x_subja to y_subja
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        #mean of row-wise min and column-wise min
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        #dother=-torch.nn.ReLU()(dother-margin) #pre 4/5/2024
        #if dother<margin, penalize (lower = more penalty).
        #if dother>=margin, ignore
        #standard triplet loss: torch.nn.ReLU()(dself-dother+margin) or torch.clamp(dself-dother+margin,min=0)
        dother=-torch.nn.ReLU()(margin-dother) #new 4/5/2024
    
    loss=dself-dother
    return loss


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
        Lz_dist = distance_loss(latent, latent, neighbor=False) # mean intersubject altent space distances should be high

        Lr = Lr_corrI + Lr_marg + (50000 * Lr_mse) # weighting MSE with 100,000 (1000 from Krakencoder)
        Lz = Lz_corrI + Lz_dist

        loss = Lr + (10 * Lz) # weighting Lz with 10 (from Krakencoder)


        mae = torch.nn.L1Loss()(pred, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss, mae


if __name__ == "__main__":
    # loads in np train data/labels
    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    # read numpy files into torch dataset and dataloader
    batch_size = 16
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=10)

    # write to file
    write_fpath = "/home/ahmadf/batch/sbatch.printKraklossTrain"
    write_to_file("Loaded in data.", filepath=write_fpath)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, eps=1e-9)

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []

    for epoch in range(1, 601):
        targets_, preds_, loss, loss2 = train(model, train_loader, device, optimizer, epoch)
        
        losses.append(float(loss.detach().numpy()))

        mae_epoch = float(loss2.detach().numpy())
        maes.append(mae_epoch)

        write_to_file(f"##### EPOCH {epoch} #####", filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN MAEs:", filepath=write_fpath) 
        write_to_file(maes, filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN Losses:", filepath=write_fpath) 
        write_to_file(losses, filepath=write_fpath)
        
        torch.save(model.state_dict(), f"/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/krakloss_{epoch}.pt")

        delete = f"/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/krakloss_{epoch-2}.pt"
        if os.path.exists(delete):
            os.remove(delete)

        
    

