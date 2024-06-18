# importing modules
from utils.dataset import *
from utils.models import *
from utils.visualize import *

import os
import os.path as op

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp

import torch

import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SplineConv
from torch_geometric.nn import GMMConv
from torch_geometric.nn import ChebConv

dataset = LOCALTranslationsData(root="/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/data/", 
                                x_type="profumo", numFeatures_x=50,
                                y_type="ICA", numFeatures_y=50)


data = dataset.get(0) # choose the subject that you want!

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = torch.device('cpu') # USE IF model=SmallestGCNConvNet()
model = GMMConvNet().to(device)
pred, model = testingForwardPass(dataset, data, model, device, model_type="GMMCONV_EPOCH")


#plotHCPSurface(cortexToSurfaceVertices(data.x.cpu().numpy())[:, 0])

#plotHCPSurface(cortexToSurfaceVertices(data.y.cpu().numpy())[:, 0])

plotHCPSurface(cortexToSurfaceVertices(pred.cpu().detach().numpy())[:, 0])
