import sys
import os
import math

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

from utils.models import *
from utils.utils import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    model_type = "ConvTransformer_SmallerDropout"
    out_nodes = 100

    torch.multiprocessing.set_sharing_strategy('file_system')

    # loads in np train/test data/labels
    test_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    # Takes only first n train and test subjects L hemis
    #train_data_np = train_data_np[:50, :, :, :]
    #train_label_np = train_label_np[:50, :]

    #test_data_np = test_data_np[:50, :, :, :]
    #test_label_np = test_label_np[:50, :]

    train_ground_truth = np.zeros(train_label_np.shape)
    train_pred = np.zeros(train_label_np.shape)
    test_ground_truth = np.zeros(test_label_np.shape)
    test_pred = np.zeros(test_label_np.shape)

    # adds start token to *_label_np
    pad=50
    test_label_np = add_start_token_np(test_label_np, n=pad)
    train_label_np = add_start_token_np(train_label_np, n=pad)


    # read numpy files into torch dataset and dataloader
    batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    write_fpath = f"/home/ahmadf/batch/temp3/sbatch.printTest2{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    # ConvTransformer
    model = ProjectionConvFullTransformer(dim_model=96, # lowkey, i think I can keep dim_model as anything I want! -- only latent_length and decoder_input_dim need compatability
                                          encoder_depth=6,
                                          nhead=6,
                                          encoder_mlp_dim=36, 
                                          decoder_input_dim=5000, 
                                          decoder_dim_feedforward=36,
                                          decoder_depth=6,
                                          dim_encoder_head=16,
                                          latent_length=100,
                                          num_channels=15,
                                          num_patches=320, 
                                          vertices_per_patch=153,
                                          dropout=0.1)


    model.load_state_dict(torch.load(f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_480.pt"))
    model.eval()
    model.to(device)

    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model params: {model_params}", filepath=write_fpath)


    train_targets = []
    test_targets = []

    train_preds_ConvDecoder = []
    test_preds_ConvDecoder = []

    train_maes = []
    test_maes = []

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)
            
            train_ground_truth[i, :] = targets.squeeze().numpy()[pad:]
            train_pred[i, :] = convDecoder_pred.squeeze().detach().numpy()[pad:]

            train_targets.append(targets)
            train_preds_ConvDecoder.append(convDecoder_pred)

            mae = np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:]))
            train_maes.append(mae)
        
        for i, data in enumerate(test_loader):
            inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)

            test_ground_truth[i, :] = targets.squeeze().numpy()[pad:]
            test_pred[i, :] = convDecoder_pred.squeeze().detach().numpy()[pad:]

            test_targets.append(targets)
            test_preds_ConvDecoder.append(convDecoder_pred)

            test_maes.append(np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])))
    
    write_to_file("TRAIN MAEs:", filepath=write_fpath)
    write_to_file(train_maes, filepath=write_fpath)
    write_to_file("TEST MAEs:", filepath=write_fpath)
    write_to_file(test_maes, filepath=write_fpath)

    #np.save(f"/home/ahmadf/NeuroTranslate/SurfToNetmat/model_output/{translation}/{model_type}/train_ground_truth.npy", train_ground_truth)
    #np.save(f"/home/ahmadf/NeuroTranslate/SurfToNetmat/model_output/{translation}/{model_type}/train_pred.npy", train_pred)
    #np.save(f"/home/ahmadf/NeuroTranslate/SurfToNetmat/model_output/{translation}/{model_type}/test_ground_truth.npy", test_ground_truth)
    #np.save(f"/home/ahmadf/NeuroTranslate/SurfToNetmat/model_output/{translation}/{model_type}/test_pred.npy", test_pred)