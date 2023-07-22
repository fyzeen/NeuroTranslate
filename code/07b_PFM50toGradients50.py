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


if __name__ == "__main__":
    # empty cache to (hopefully) help memory
    torch.cuda.empty_cache()

    # initialize the dataset
    dataset = TranslationsData(x_type="profumo", numFeatures_x=50,
                               y_type="gradients", numFeatures_y=50)
    
    
    # split the dataset into train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) 

    # load data
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    
    # initialize the model on the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GMMConvNet().to(device)

    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.SmoothL1Loss()

    def train(epoch):
        model.train()
        
        if epoch == 100: ###### THIS WAS INITIALLY 100 #######
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.005

        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()

            loss = torch.nn.SmoothL1Loss()
            output = model(data)
            output_loss = loss(output, data.y)
            output_loss.backward()

            MAE = torch.mean(abs(data.to(device).y - model(data))).item()  # Checking performance
            optimizer.step()
        
        return output_loss.detach(), MAE

    def test():
        model.eval()

        MeanAbsError = 0
        y = []
        y_hat = []

        for data in test_loader:
            with torch.no_grad():
                pred = model(data.to(device)).detach()
            y_hat.append(pred)
            y.append(data.to(device).y)

            MAE = torch.mean(abs(data.to(device).y - model(data))).item()  # Checking performance
            MeanAbsError += MAE

        test_MAE = MeanAbsError / len(test_loader)
        output = {'Predicted_values': y_hat, 'Measured_values': y, 'MAE': test_MAE}
        return output
    
    def write_to_file(content):
        with open('/home/ahmadf/batch/sbatch.print6', 'a') as file:
            file.write(str(content) + '\n')


    # print subjects in train and test
    print("### TRAIN SUBJECTS ###")
    print([dataset.subj_list[i] for i in train_dataset.indices])
    print("### TEST SUBJECTS ###")
    print([dataset.subj_list[i] for i in test_dataset.indices])
    write_to_file("### TRAIN SUBJECTS ###")
    write_to_file([dataset.subj_list[i] for i in train_dataset.indices])
    write_to_file("### TEST SUBJECTS ###")
    write_to_file([dataset.subj_list[i] for i in test_dataset.indices])
    
    train_losses = []
    train_MAEs = []
    test_MAEs = []

    for epoch in range(1, 20):
        loss, MAE = train(epoch)
        test_output = test()

        train_losses.append(loss)
        train_MAEs.append(MAE)
        test_MAEs.append(test_output['MAE'])

        print(f"EPOCH: {epoch}, Train_loss: {loss}, Train_MAE: {MAE}, Test_MAE: {test_output['MAE']}")
        write_to_file(f"EPOCH: {epoch}, Train_loss: {loss}, Train_MAE: {MAE}, Test_MAE: {test_output['MAE']}")

        torch.save(model.state_dict(), op.join("/scratch/ahmadf/NeuroTranslate/saved_models/profumo_d50_to_gradients_d50/", f"GMMCONV_recentEPOCH_profumo_d50_to_gradients_d50.pt")) 

        if epoch % 20 == 0:
            torch.save(model.state_dict(), op.join("/scratch/ahmadf/NeuroTranslate/saved_models/profumo_d50_to_gradients_d50/", f"GMMCONV_EPOCH{epoch}_profumo_d50_to_gradients_d50.pt")) 
        
    print("#############################")
    print("##### TRAINING COMPLETE #####")
    print("#############################")
    print("Train Losses:")
    print(train_losses)
    print("#############################")
    print("Train_MAEs:")
    print(train_MAEs)
    print("#############################")
    print("Test_MAEs:")
    print(test_MAEs)
    print("#############################")

    torch.save(model.state_dict(), op.join("/scratch/ahmadf/NeuroTranslate/saved_models/profumo_d50_to_gradients_d50/", "GMMCONVV_FINAL_profumo_d50_to_gradients_d50.pt")) 
