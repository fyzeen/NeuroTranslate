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

def train(model, train_loader, loss_fn, device, input_dim, optimizer, epoch, reset_params=True):
    model.train()

    targets_ = []
    preds_ = []

    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) # USE THIS unsqueeze(0) ONLY if batch size = 1
        
        pred = model(src=inputs, tgt=targets, tgt_mask=generate_subsequent_mask(model.latent_length).to(device))

        loss = loss_fn(pred, targets)

        loss2func = torch.nn.L1Loss()
        loss2 = loss2func(pred, targets)

        loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        targets_.append(targets.cpu().numpy())
        preds_.append(pred.cpu().detach().numpy())

    return targets_, preds_, loss, loss2


if __name__ == "__main__":
    translation = "ICAd15_schfd100"
    model_type = "GraphTransformer"
    out_nodes = 100

    # loads in np train data/labels
    train_data_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_label_np = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    # make netmat and add start node(s) -- you need to have an non-prime number of nodes so that each attetion head is the same size
    train_label_np = make_nemat_allsubj(train_label_np, out_nodes)

    num_start_nodes = 2
    for i in range(num_start_nodes):
        train_label_np = add_start_node(train_label_np)

    # read numpy files into torch dataset and dataloader
    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=10)

    # write to file
    write_fpath = f"/home/ahmadf/batch/sbatch.print{model_type}_{translation}"
    write_to_file("Loaded in data.", filepath=write_fpath)


    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"


    # GraphTransformer
    model = GraphTransformer(dim_model=102, 
                             encoder_depth=6, 
                             nhead=6, 
                             encoder_mlp_dim=102,
                             decoder_input_dim=102, 
                             decoder_dim_feedforward=102,
                             decoder_depth=6,
                             dim_encoder_head=17,
                             latent_length=102,
                             num_channels=15,
                             num_patches=320,
                             vertices_per_patch=153,
                             dropout=0.1)
    
    # initialize optimizer / loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001, eps=1e-9)
    loss_fn = torch.nn.MSELoss()

    # reset params 
    model._reset_parameters()

    maes = []
    losses = []

    for epoch in range(1, 601):
        targets_, preds_, loss, loss2 = train(model, train_loader, loss_fn, device, model.input_dim, optimizer, epoch)
        
        losses.append(float(loss.detach().numpy()))

        #mae_epoch = np.mean(np.abs(np.concatenate(targets_) - np.concatenate(preds_)))
        mae_epoch = float(loss2.detach().numpy())
        maes.append(mae_epoch)

        write_to_file(f"##### EPOCH {epoch} #####", filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN MAEs:", filepath=write_fpath) 
        write_to_file(maes, filepath=write_fpath)

        write_to_file(f"EPOCH 1-{epoch} TRAIN Losses:", filepath=write_fpath) 
        write_to_file(losses, filepath=write_fpath)
        
        torch.save(model.state_dict(), f"/home/ahmadf/NeuroTranslate/SurfToNetmat/saved_models/{translation}/{model_type}_{epoch}.pt")

        delete = f"/home/ahmadf/NeuroTranslate/SurfToNetmat/TransformerTest/saved_models/{translation}/{model_type}_{epoch-2}.pt"
        if os.path.exists(delete):
            os.remove(delete)
