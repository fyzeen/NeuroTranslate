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

dataset = LOCALTranslationsData(root = "/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/data/",
                                x_type="ICA", numFeatures_x=50,
                                y_type="profumo", numFeatures_y=50)

subj_idx = 0
data = dataset.get(subj_idx) # choose the subject that you want!

pred = np.load(f"/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/saved_models/ICA_d50_to_profumo_d50/shallowSPLINECONV_OUT_subj{subj_idx}.npy")

plotHCPSurface(cortexToSurfaceVertices(data.x.cpu().numpy())[:, 23])

plotHCPSurface(cortexToSurfaceVertices(data.y.cpu().numpy())[:, 0])

#plotHCPSurface(cortexToSurfaceVertices(pred)[:, 10])
