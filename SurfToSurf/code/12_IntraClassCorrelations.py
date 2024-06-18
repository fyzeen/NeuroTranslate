# importing modules
from utils.dataset import *
from utils.models import *
from utils.visualize import *
from utils.compute import *

import os
import os.path as op

from munkres import Munkres

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp
import pandas as pd
import pingouin as pg

import torch
import torch.nn.functional as F

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def write_to_file(content):
    with open('/home/ahmadf/batch/sbatch.print27', 'a') as file:
        file.write(str(content) + '\n')

translations = [["ICA", 50, "profumo", 50],
                ["ICA", 50, "gradients", 50],
                ["gradients", 50, "profumo", 50],
                ["ICA", 15, "profumo", 50],
                ["ICA", 200, "profumo", 50],
                ["profumo", 50, "ICA", 50],
                ["profumo", 50, "gradients", 50],
                ["gradients", 50, "ICA", 50],
                ["ICA", 200, "gradients", 50],
                ["gradients", 200, "profumo", 50]]

# Random sample of subjects; first ten in train set, second ten in test set
subj_list = [107422, 175136, 149337, 188549, 125222, 135225, 181232, 119025, 809252, 134021,
             334635, 136227, 481951, 285345, 214625, 329440, 116726, 178849, 181131, 381038]

for i in [8]:
    # Read dataset for correct translation
    dataset = TranslationsData(x_type=translations[i][0], numFeatures_x=translations[i][1],
                               y_type=translations[i][2], numFeatures_y=translations[i][3])
    
    # Select the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    for model, model_type in [(LargerGCNConvNet().to(device), "largerGCNCONV_EPOCH"),
                              (GMMConvNet().to(device), "GMMCONV_EPOCH"), 
                              (SplineConvNet().to(device), "SPLINECONV_EPOCH")]: 
    '''
    for model, model_type in [(SplineConvNet().to(device), "SPLINECONV_EPOCH")]:
        
        # Initialize the output DataFrame
        out_df = pd.DataFrame(columns=['Subject ID'])
        for j in range(translations[i][3]):
            icc_column_name = f'SubjectICC_dim{i}'
            out_df[icc_column_name] = None

        # Loop through subjects
        for num, j in enumerate(subj_list): #range(1003)
            data = dataset.get(j)
            if num < 10:
                subject_ID = "train"+str(j) #int(dataset.subj_list[j])
            else:
                subject_ID = "test"+str(j)

            # Do forward pass
            pred, model = testingForwardPass(dataset, data, model, device, model_type=model_type)

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

            ICCs = []
            for k in range(translations[i][3]):
                # Compute the ICC (ratings=column_number (i.e., dimension of the output you want to do an ICC on))
                results = pg.intraclass_corr(data=df, targets='vertex', raters='type', ratings=k)
                icc = results.loc[2]["ICC"]
                ICCs.append(icc)

            print(subject_ID, ICCs)
            write_to_file(f"{subject_ID} {ICCs}")
            # Append values to the DataFrame
            to_append = {'Subject ID': subject_ID}
            for k, dim_n_ICC in enumerate(ICCs):
                icc_column_name = f'SubjectICC_dim{k}'
                to_append[icc_column_name] = dim_n_ICC
            out_df = out_df.append(to_append, ignore_index=True)
        
        out_df.to_csv(f"/scratch/ahmadf/NeuroTranslate/model_ICCs/0{i+1}_{translations[i][0]}{translations[i][1]}to{translations[i][2]}{translations[i][3]}_{model_type}_ICCs.csv")
        print("################")
        write_to_file("################")

    if model_type == "largerGCNCONV_EPOCH":
        # Now we compute ICCs for hungarian algorithm solution
        out_df = pd.DataFrame(columns=['Subject ID', 'Subject ICC'])
        for num, j in enumerate(subj_list): #range(1003)
            data = dataset.get(j)
            if num < 10:
                subject_ID = "train"+str(j) #int(dataset.subj_list[j])
            else:
                subject_ID = "test"+str(j)

            dists = cosine_distance_columns(data.x, data.y)

            m = Munkres()
            assignments = m.compute(dists.tolist())

            pred = reorder_columns(data.x, data.y, assignments)

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
            results = pg.intraclass_corr(data=df, targets='vertex', raters='type', ratings=0)

            icc = results.loc[2]["ICC"]
            print(subject_ID, icc)
            write_to_file(f"{subject_ID} {icc}")

            # Append values to the DataFrame
            out_df = out_df.append({'Subject ID': subject_ID, 'Subject ICC': icc}, ignore_index=True)
        
        out_df.to_csv(f"/scratch/ahmadf/NeuroTranslate/model_ICCs/0{i+1}_{translations[i][0]}{translations[i][1]}to{translations[i][2]}{translations[i][3]}_HUNGARIAN_ICCs.csv")
        print("################")
        write_to_file("################")

