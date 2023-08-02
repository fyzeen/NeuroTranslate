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
import pandas as pd
import pingouin as pg

import torch
import torch.nn.functional as F


dataset = LOCALTranslationsData(root="/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/data/", 
                                x_type="ICA", numFeatures_x=50,
                                y_type="gradients", numFeatures_y=50)

# Read a subject
data = dataset.get(0)

# Do a forward pass on your model to get prediction
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = GMMConvNet().to(device)
pred, model = testingForwardPass(dataset, data, model, device, model_type="GMMCONV_EPOCH")

# Convert ground truth and predictions to numpy arrays (shape: [59412, nDim])
real = data.y.cpu().numpy()
np_pred = pred.cpu().numpy()

# Convert numpy arrays to pandas DataFrames
real_df = pd.DataFrame(real)
pred_df = pd.DataFrame(np_pred)

# Label each type (ground_truth or prediction), label each vertex as a "group" for the ICC
real_df["type"] = "ground_truth"
real_df["vertex"] = np.arange(0, 59412, 1)
pred_df["type"] = "pred"
pred_df["vertex"] = np.arange(0, 59412, 1)

# Vertically concatenate the dataframes
df = pd.concat([real_df, pred_df])

# Compute the ICC (ratings=column_number (i.e., dimension of the output you want to do an ICC on))
results = pg.intraclass_corr(data=df, targets='vertex', raters='type', ratings=1)

# Show results
print(results)
