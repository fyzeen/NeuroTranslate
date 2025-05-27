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


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_printoptions(threshold=10_000)

    # loads in np train/test data/labels
    test_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/test_data.npy")
    test_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/test_labels.npy")

    test_ground_truth = np.zeros(test_label_np.shape)
    test_pred = np.zeros(test_label_np.shape)


    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    train_ground_truth = np.zeros(train_label_np.shape)
    train_pred = np.zeros(train_label_np.shape)

    # read numpy files into torch dataset and dataloader
    batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    # write to file
    write_fpath = "/home/ahmadf/batch/sbatch.printKraklossTest"
    write_to_file("Loaded in data.", filepath=write_fpath)


    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # SiTNoPool
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
    
    model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/krakloss_600.pt"))

    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model params: {model_params}", filepath=write_fpath)

    # Testing below

    model.eval()
    model.to(device)

    mse_train_list = []
    mae_train_list = []

    mse_test_list = []
    mae_test_list = []

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_train_list.append(mae)

            mse = np.mean( (targets.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_train_list.append(mse)

            train_ground_truth[i, :] = targets.squeeze().numpy()
            train_pred[i, :] = pred.squeeze().detach().numpy()
        
        for i, data in enumerate(test_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            pred, latent = model(img=inputs)

            mae = np.mean(np.abs(targets.squeeze().numpy() - pred.squeeze().detach().numpy()))
            mae_test_list.append(mae)

            mse = np.mean( (targets.squeeze().numpy() - pred.squeeze().detach().numpy())**2 )
            mse_test_list.append(mse)

            test_ground_truth[i, :] = targets.squeeze().numpy()
            test_pred[i, :] = pred.squeeze().detach().numpy()

    write_to_file("TRAIN Mean MAE:", filepath=write_fpath)
    write_to_file(np.mean(mae_train_list), filepath=write_fpath)
    write_to_file("TEST Mean MAE:", filepath=write_fpath)
    write_to_file(np.mean(mae_test_list), filepath=write_fpath)

    write_to_file("TRAIN Mean MSE:", filepath=write_fpath)
    write_to_file(np.mean(mse_train_list), filepath=write_fpath)
    write_to_file("TEST Mean MSE:", filepath=write_fpath)
    write_to_file(np.mean(mse_test_list), filepath=write_fpath)

    np.save("/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/KrakLossModel/train_ground_truth.npy", train_ground_truth)
    np.save("/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/KrakLossModel/train_pred.npy", train_pred)
    np.save("/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/KrakLossModel/test_ground_truth.npy", test_ground_truth)
    np.save("/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/KrakLossModel/test_pred.npy", test_pred)






    

        
    

