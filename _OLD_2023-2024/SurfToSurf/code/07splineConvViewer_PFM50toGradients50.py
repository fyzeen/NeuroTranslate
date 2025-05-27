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

dataset = TranslationsData(x_type="profumo", numFeatures_x=50,
                           y_type="gradients", numFeatures_y=50)

subj_idx = 0
data = dataset.get(subj_idx) # choose the subject that you want!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SplineConvNet().to(device)
pred, model = testingForwardPass(dataset, data, model, device, model_type="SPLINECONV_EPOCH")

out = pred.cpu().detach().numpy()

np.save(f"/scratch/ahmadf/NeuroTranslate/saved_models/profumo_d50_to_gradients_d50/SPLINECONV_OUT_subj{subj_idx}.npy", out)

