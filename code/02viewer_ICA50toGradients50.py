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
                                x_type="ICA", numFeatures_x=50,
                                y_type="gradients", numFeatures_y=50)


data = dataset.get(0) # choose the subject that you want!

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = torch.device('cpu') # USE IF model=SmallestGCNConvNet()
model = GMMConvNet().to(device)
model.load_state_dict(torch.load("/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/saved_models/ICA_d50_to_gradients_d50/GMMCONV_EPOCH55_ICA_d50_to_gradients_d50.pt", map_location=device))
model.eval()

with torch.no_grad():
    pred = model(data.to(device)).to(device)

#plotHCPSurface(cortexToSurfaceVertices(data.x.cpu().numpy())[:, 0])

plotHCPSurface(cortexToSurfaceVertices(data.y.cpu().numpy())[:, 1])

plotHCPSurface(cortexToSurfaceVertices(pred.cpu().detach().numpy())[:, 1])
