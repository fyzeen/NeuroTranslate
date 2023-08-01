# importing modules
from utils.dataset import *
from utils.models import *
from utils.compute import *

import os
import os.path as op

from munkres import Munkres

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp

import torch
import torch.nn.functional as F

dataset = TranslationsData(x_type="ICA", numFeatures_x=200,
                           y_type="gradients", numFeatures_y=50)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

mae_list = []
total_mae = 0

print("### ASSIGNMENTS ###")
i = 0
for data in loader:
    dists = cosine_distance_columns(data.x, data.y)

    m = Munkres()
    assignments = m.compute(dists.tolist())

    print(f"SUBJ ID {i}")
    print(assignments)

    pred = reorder_columns(data.x, data.y, assignments)

    MAE = torch.mean(abs(data.y - pred)).item()

    mae_list.append(MAE)
    total_mae += MAE
    i += 1

out_MAE = total_mae / len(loader)

print("### List of all MAEs ###")
print(mae_list)

print("### Avg. MAE ###")
print(out_MAE)
