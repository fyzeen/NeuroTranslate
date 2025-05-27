from models import *
from utils import *

import os
import sys

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # loads in np train data/labels
    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    write_fpath = "/home/ahmadf/batch/sbatch.printForwardPass_ConvModel"

    write_to_file("Loaded in data.", filepath=write_fpath)

    # takes only the first L hemi
    train_data_np = train_data_np[:1, :, :, :]
    train_label_np = train_label_np[:1, :]

    # adds start token to train_labl_np
    train_label_np = add_start_token_np(train_label_np)

    # read numpy files into torch dataset and dataloader
    batch_size = 1
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    model = ProjectionConvFullTransformer(dim_model=192, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
                                          encoder_depth=3,
                                          nhead=6,
                                          encoder_mlp_dim=36, 
                                          decoder_input_dim=5000, 
                                          decoder_dim_feedforward=36,
                                          decoder_depth=6,
                                          dim_encoder_head=6,
                                          latent_length=50,
                                          dropout=0.1)

    model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/SMALLConv_Model44.pt"))
    model.eval()
    model.to(device)

    targets_ = []
    preds_ = []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0)
            print(targets.shape)

            predictions = greedy_decode(model=model, source=inputs, input_dim=5000, device=device, b=batch_size)

            targets_.append(targets)
            preds_.append(predictions)

    torch.set_printoptions(threshold=10_000)
    
    write_to_file("TARGETS:", filepath=write_fpath)
    write_to_file(targets_, filepath=write_fpath)
    write_to_file("PREDS:", filepath=write_fpath)
    write_to_file(preds_, filepath=write_fpath)
