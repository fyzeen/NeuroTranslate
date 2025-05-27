# importing modules
from utils.dataset import *
from utils.models import *
from utils.visualize import *

import os
import os.path as op

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import pandas as pd
import pingouin as pg
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

dataset = LOCALTranslationsData(root = "/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/SurfToSurf/data/",
                                x_type="gradients", numFeatures_x=50,
                                y_type="ICA", numFeatures_y=50)


subj_idx = 0
data = dataset.get(subj_idx) # choose the subject that you want!

pred = np.load(f"/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/SurfToSurf/saved_models/gradients_d50_to_ICA_d50/SPLINECONV_OUT_subj{subj_idx}.npy")

#plotHCPSurface(cortexToSurfaceVertices(data.x.cpu().numpy())[:, 0])

#plotHCPSurface(cortexToSurfaceVertices(data.y.cpu().numpy())[:, 4])

#plotHCPSurface(cortexToSurfaceVertices(pred)[:, 4])


# Convert numpy arrays to pandas DataFrames
real = data.y.cpu().numpy()[:, 4]
np_pred = pred[:, 4]
real_df = pd.DataFrame(real)
pred_df = pd.DataFrame(np_pred)

# Label each type (ground_truth or prediction), label each vertex as a "group" for the ICC
real_df["type"] = "ground_truth"
real_df["vertex"] = np.arange(0, 59412, 1)
pred_df["type"] = "pred"
pred_df["vertex"] = np.arange(0, 59412, 1)

# Vertically concatenate the dataframes
df = pd.concat([real_df, pred_df])
results = pg.intraclass_corr(data=df, targets='vertex', raters='type', ratings=0)
icc = results.loc[2]["ICC"]
print(results)
print(icc)
