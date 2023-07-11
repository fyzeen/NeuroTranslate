# importing modules
from utils.dataset import *
from utils.models import *

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

class LOCALTranslationsData(Dataset):
    def __init__(self, root, x_type=None, y_type=None, numFeatures_x=None, numFeatures_y=None, brain_shape="inflated",
                 transform=None, pre_transform=None, pre_filter=None):
        
        self.root = root
        
        self.x_type = x_type
        self.y_type = y_type
        self.numFeatures_x = numFeatures_x
        self.numFeatures_y = numFeatures_y
        self.brain_shape = brain_shape
        
        self.x_raw_path = op.join(root, "raw", f"{x_type}_d{numFeatures_x}")
        self.y_raw_path = op.join(root, "raw", f"{y_type}_d{numFeatures_y}")
                        
        self.subj_list = list(np.genfromtxt(op.join(root, "LOCAL_subj_list.csv"), delimiter=",", dtype=int))
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return {'x':sorted(list(ListDirNoHidden(self.x_raw_path))), 'y':sorted(list(ListDirNoHidden(self.y_raw_path)))}
    
    @property
    def processed_file_names(self):
        return sorted(list(ListDirNoHidden(self.processed_dir)))
    
    @property
    def processed_dir(self):
        return op.join(self.root, "processed", f"{self.x_type}_d{self.numFeatures_x}_TO_{self.y_type}_d{self.numFeatures_y}")
    

    def process(self):
        for i, x_fname in enumerate(self.raw_file_names['x']):
            y_fname = self.raw_file_names['y'][i]
            
            x_path = op.join(self.x_raw_path, x_fname)
            y_path = op.join(self.y_raw_path, y_fname)
            
            subjData = read_Subj(x=x_path, y=y_path, 
                                 numFeatures_x=self.numFeatures_x, numFeatures_y=self.numFeatures_y, 
                                 brain_shape=self.brain_shape)
                                 
            torch.save(subjData, op.join(self.processed_dir, f"subj{self.subj_list[i]}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(op.join(self.processed_dir, f"subj{self.subj_list[idx]}.pt"))


dataset = LOCALTranslationsData(root="/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/data/", 
                x_type="ICA", numFeatures_x=50,
                y_type="profumo", numFeatures_y=50)

dataloader = DataLoader(dataset, batch_size=1)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model=GCNConvNet().to(device)
model.load_state_dict(torch.load("/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/saved_models/ICA_d50_to_profumo_d50/GCNCONV_EPOCH4_ICA_d50_to_profumo_d50.pt", map_location=device))
model.eval()

with torch.no_grad():
    pred = model(next(iter(dataloader)).to(device))

surf1 = plotting.view_surf(hcp.mesh.inflated, pred.cpu().detach().numpy()[:, 0], threshold=0, bg_map=hcp.mesh.sulc)
surf1.open_in_browser()

surf2 = plotting.view_surf(hcp.mesh.inflated, next(iter(dataloader)).y.numpy()[:, 0], threshold=0, bg_map=hcp.mesh.sulc)
surf2.open_in_browser()

surf3 = plotting.view_surf(hcp.mesh.inflated, next(iter(dataloader)).x.numpy()[:, 0], threshold=0, bg_map=hcp.mesh.sulc)
surf3.open_in_browser()



