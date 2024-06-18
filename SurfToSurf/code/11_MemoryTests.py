# importing modules
from utils.dataset import *
from utils.models import *

import os
import os.path as op
import time

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
    dataset = TranslationsData(x_type="ICA", numFeatures_x=50,
                               y_type="profumo", numFeatures_y=50)
    
    
    # split the dataset into train and test sets
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size]) 

    def train(model, optimizer):
        model.train()

        # Reset the maximum memory usage tracker
        torch.cuda.reset_max_memory_allocated(device=None)

        try:
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()

                loss = torch.nn.SmoothL1Loss()
                output = model(data)
                output_loss = loss(output, data.y)
                output_loss.backward()

                MAE = torch.mean(abs(data.to(device).y - model(data))).item()  # Checking performance
                optimizer.step()

        except RuntimeError as e:
            # Catch CUDA out of memory error
            if 'out of memory' in str(e):
                write_to_file(e)
                print(e)
                torch.cuda.empty_cache()
                # Calculate the maximum memory used during training
                max_memory_used = torch.cuda.max_memory_allocated()
                return None, None, max_memory_used, True

            else:
                # Re-raise other RuntimeError (if any other RuntimeError occurs)
                raise e
            
        # Calculate the maximum memory used during training
        max_memory_used = torch.cuda.max_memory_allocated()

        return output_loss.detach(), MAE, max_memory_used, False

    def test(model):
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
        with open('/home/ahmadf/batch/sbatch.print', 'a') as file:
            file.write(str(content) + '\n')

    for i in [1, 3, 5, 10, 20]:
        # load data
        train_loader = DataLoader(train_dataset, batch_size=i, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=i, shuffle=False)

        for j, item in enumerate([LargerGCNConvNet(), ShallowGCNConvNet(), GMMConvNet(), ShallowGMMConvNet(), SplineConvNet(), ShallowSplineConvNet()]):
            torch.cuda.empty_cache()

            model_name_list = ["LargerGCNConvNet", "ShallowGCNConvNet", "GMMConvNet", "ShallowGMMConvNet", "SplineConvNet", "ShallowSplineConvNet"]
            write_to_file(f"########## TESTING: {model_name_list[j]} with batch size {i} ##########")
            print(f"########## TESTING: {model_name_list[j]} with batch size {i} ##########")

            # initialize the model on the GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = item.to(device)

            # initialize optimizer / loss
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            loss = torch.nn.SmoothL1Loss()
            
            train_losses = []
            train_MAEs = []
            test_MAEs = []

            epoch = 1
            while epoch < 6:
                start_time = time.time()
                loss, MAE, max_memory_used, stop = train(model, optimizer)
                if stop:
                    epoch = 6
                
                if epoch < 6: 
                    test_output = test(model)

                    train_losses.append(loss)
                    train_MAEs.append(MAE)
                    test_MAEs.append(test_output['MAE'])

                    print(f"EPOCH: {epoch}, Train_loss: {loss}, Train_MAE: {MAE}, Test_MAE: {test_output['MAE']}")
                    print(f"MAX MEM USED: {max_memory_used / 1024**3:.4f} GB")
                    print(f"Time for epoch {epoch}: {(time.time() - start_time) / 60:.4f} min")
                    write_to_file(f"EPOCH: {epoch}, Train_loss: {loss}, Train_MAE: {MAE}, Test_MAE: {test_output['MAE']}")
                    write_to_file(f"MAX MEM USED: {max_memory_used / 1024**3:.4f} GB")
                    write_to_file(f"Time for epoch {epoch}: {(time.time() - start_time) / 60:.4f} min")
                    epoch += 1

                else: 
                    write_to_file("########## FINISHED TRAINING ##########")
                    print("########## FINISHED TRAINING ##########")
                    write_to_file(f"MAX MEM USED: {max_memory_used / 1024**3:.4f} GB")
                    print(f"MAX MEM USED: {max_memory_used / 1024**3:.4f} GB")
                    write_to_file(f"Time for epoch {epoch}: {(time.time() - start_time) / 60:.4f} min")
                    print(f"Time for epoch {epoch}: {(time.time() - start_time) / 60:.4f} min")
                    write_to_file("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                    del model
                    del optimizer
                    del train_losses
                    del train_MAEs
                    del test_MAEs
                    torch.cuda.empty_cache()
