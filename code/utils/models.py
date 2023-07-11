# importing modules
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


class SplineConvNet(torch.nn.Module):
    def __init__(self):
        super(SplineConvNet, self).__init__()
        
        self.conv1 = SplineConv(50, 75, dim=3, kernel_size=25, norm=False).float()
        self.bn1 = torch.nn.BatchNorm1d(75)
        
        self.conv2 = SplineConv(75, 100, dim=3, kernel_size=25, norm=False).float()
        self.bn2 = torch.nn.BatchNorm1d(100)
        
        self.conv3 = SplineConv(100, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn3 = torch.nn.BatchNorm1d(200)
        
        self.conv4 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn4 = torch.nn.BatchNorm1d(200)
       
        self.conv5 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn5 = torch.nn.BatchNorm1d(200)

        self.conv6 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn6 = torch.nn.BatchNorm1d(200)

        self.conv7 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn7 = torch.nn.BatchNorm1d(200)

        self.conv8 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn8 = torch.nn.BatchNorm1d(200)

        self.conv9 = SplineConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn9 = torch.nn.BatchNorm1d(200)
        
        self.conv10 = SplineConv(200, 100, dim=3, kernel_size=25, norm=False).float()
        self.bn10 = torch.nn.BatchNorm1d(100)
        
        self.conv11 = SplineConv(100, 75, dim=3, kernel_size=25, norm=False).float()
        self.bn11 = torch.nn.BatchNorm1d(75)
        
        self.conv12 = SplineConv(75, 50, dim=3, kernel_size=25, norm=False).float()
            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.elu(self.conv1(x, edge_index, edge_attr)) 
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv9(x, edge_index, edge_attr))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv10(x, edge_index, edge_attr))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index, edge_attr))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = self.conv12(x, edge_index, edge_attr)
        
        return x
    
class GCNConvNet(torch.nn.Module):
    def __init__(self):
        super(GCNConvNet, self).__init__()
        
        self.conv1 = GCNConv(-1, 100).float()
        self.bn1 = torch.nn.BatchNorm1d(100)
        
        self.conv2 = GCNConv(100, 200).float()
        self.bn2 = torch.nn.BatchNorm1d(200)
        
        self.conv3 = GCNConv(200, 400).float()
        self.bn3 = torch.nn.BatchNorm1d(400)
        
        self.conv4 = GCNConv(400, 400).float()
        self.bn4 = torch.nn.BatchNorm1d(400)

        self.conv5 = GCNConv(400, 400).float()
        self.bn5 = torch.nn.BatchNorm1d(400)

        self.conv6 = GCNConv(400, 400).float()
        self.bn6 = torch.nn.BatchNorm1d(400)

        self.conv7 = GCNConv(400, 400).float()
        self.bn7 = torch.nn.BatchNorm1d(400)

        self.conv8 = GCNConv(400, 400).float()
        self.bn8 = torch.nn.BatchNorm1d(400)

        self.conv9 = GCNConv(400, 400).float()
        self.bn9 = torch.nn.BatchNorm1d(400)
        
        self.conv10 = GCNConv(400, 200).float()
        self.bn10 = torch.nn.BatchNorm1d(200)
        
        self.conv11 = GCNConv(200, 100).float()
        self.bn11 = torch.nn.BatchNorm1d(100)
        
        self.conv12 = GCNConv(100, 50).float()
            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.elu(self.conv1(x, edge_index)) 
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training) 

        x = F.elu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training) 

        x = F.elu(self.conv7(x, edge_index))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)   

        x = F.elu(self.conv8(x, edge_index))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)   

        x = F.elu(self.conv9(x, edge_index))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)                  
    
        x = F.elu(self.conv10(x, edge_index))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = self.conv12(x, edge_index)
        
        return x
    

class GMMConvNet(torch.nn.Module):
    def __init__(self):
        super(GMMConvNet, self).__init__()
        
        self.conv1 = GMMConv(50, 75, dim=3, kernel_size=25, norm=False).float()
        self.bn1 = torch.nn.BatchNorm1d(75)
        
        self.conv2 = GMMConv(75, 100, dim=3, kernel_size=25, norm=False).float()
        self.bn2 = torch.nn.BatchNorm1d(100)
        
        self.conv3 = GMMConv(100, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn3 = torch.nn.BatchNorm1d(200)
        
        self.conv4 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn4 = torch.nn.BatchNorm1d(200)

        self.conv5 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn5 = torch.nn.BatchNorm1d(200)

        self.conv6 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn6 = torch.nn.BatchNorm1d(200)

        self.conv7 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn7 = torch.nn.BatchNorm1d(200)

        self.conv8 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn8 = torch.nn.BatchNorm1d(200)

        self.conv9 = GMMConv(200, 200, dim=3, kernel_size=25, norm=False).float()
        self.bn9 = torch.nn.BatchNorm1d(200)
        
        self.conv10 = GMMConv(200, 100, dim=3, kernel_size=25, norm=False).float()
        self.bn10 = torch.nn.BatchNorm1d(100)
        
        self.conv11 = GMMConv(100, 75, dim=3, kernel_size=25, norm=False).float()
        self.bn11 = torch.nn.BatchNorm1d(75)
        
        self.conv12 = GMMConv(75, 50, dim=3, kernel_size=25, norm=False).float()
            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.elu(self.conv1(x, edge_index, edge_attr)) 
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index, edge_attr))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv4(x, edge_index, edge_attr))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)   

        x = F.elu(self.conv5(x, edge_index, edge_attr))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)  

        x = F.elu(self.conv6(x, edge_index, edge_attr))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)   

        x = F.elu(self.conv7(x, edge_index, edge_attr))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index, edge_attr))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)  

        x = F.elu(self.conv9(x, edge_index, edge_attr))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)  
        
        x = F.elu(self.conv10(x, edge_index, edge_attr))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index, edge_attr))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = self.conv12(x, edge_index, edge_attr)
        
        return x
    
class ChebConvNet(torch.nn.Module):
    def __init__(self):
        super(ChebConvNet, self).__init__()
        
        self.conv1 = ChebConv(-1, 100, K=4).float()
        self.bn1 = torch.nn.BatchNorm1d(100)
        
        self.conv2 = ChebConv(100, 200, K=4).float()
        self.bn2 = torch.nn.BatchNorm1d(200)
        
        self.conv3 = ChebConv(200, 400, K=4).float()
        self.bn3 = torch.nn.BatchNorm1d(400)
        
        self.conv4 = ChebConv(400, 400, K=4).float()
        self.bn4 = torch.nn.BatchNorm1d(400)

        self.conv5 = ChebConv(400, 400, K=4).float()
        self.bn5 = torch.nn.BatchNorm1d(400)

        self.conv6 = ChebConv(400, 400, K=4).float()
        self.bn6 = torch.nn.BatchNorm1d(400)

        self.conv7 = ChebConv(400, 400, K=4).float()
        self.bn7 = torch.nn.BatchNorm1d(400)

        self.conv8 = ChebConv(400, 400, K=4).float()
        self.bn8 = torch.nn.BatchNorm1d(400)

        self.conv9 = ChebConv(400, 400, K=4).float()
        self.bn9 = torch.nn.BatchNorm1d(400)
        
        self.conv10 = ChebConv(400, 200, K=4).float()
        self.bn10 = torch.nn.BatchNorm1d(200)
        
        self.conv11 = ChebConv(200, 100, K=4).float()
        self.bn11 = torch.nn.BatchNorm1d(100)
        
        self.conv12 = ChebConv(100, 50, K=4).float()
            
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = F.elu(self.conv1(x, edge_index)) 
        x = self.bn1(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv6(x, edge_index))
        x = self.bn6(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv7(x, edge_index))
        x = self.bn7(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv8(x, edge_index))
        x = self.bn8(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv9(x, edge_index))
        x = self.bn9(x)
        x = F.dropout(x, p=.10, training=self.training)
        
        x = F.elu(self.conv10(x, edge_index))
        x = self.bn10(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = F.elu(self.conv11(x, edge_index))
        x = self.bn11(x)
        x = F.dropout(x, p=.10, training=self.training)

        x = self.conv12(x, edge_index)
        
        return x