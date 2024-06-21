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
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_printoptions(threshold=10_000)

    # loads in np train/test data/labels
    test_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/test_data.npy")
    test_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/test_labels.npy")

    train_data_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_data.npy")
    train_label_np = np.load("/scratch/naranjorincon/surface-vision-transformers/data/ICAd15_schfd100/template/train_labels.npy")

    # Takes only first n train and test subjects L hemis
    #n=200
    #train_data_np = train_data_np[:5, :, :, :]
    #train_label_np = train_label_np[:5, :]

    #test_data_np = test_data_np[:5, :, :, :]
    #test_label_np = test_label_np[:5, :]

    in_train_label_np = make_nemat_allsubj(train_label_np, 100)
    in_train_label_np = add_start_node(in_train_label_np)
    in_train_label_np = add_start_node(in_train_label_np)

    in_test_label_np = make_nemat_allsubj(test_label_np, 100)
    in_test_label_np = add_start_node(in_test_label_np)
    in_test_label_np = add_start_node(in_test_label_np)


    # Creates out arrays
    train_ground_truth = np.zeros(train_label_np.shape)
    train_pred = np.zeros(train_label_np.shape)

    test_ground_truth = np.zeros(test_label_np.shape)
    test_pred = np.zeros(test_label_np.shape)

    # read numpy files into torch dataset and dataloader
    batch_size = 1
    test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_data_np).float(), torch.from_numpy(test_label_np).float(), torch.from_numpy(in_test_label_np).float())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data_np).float(), torch.from_numpy(train_label_np).float(), torch.from_numpy(in_train_label_np).float())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle=False, num_workers=10)

    write_fpath = "/home/ahmadf/batch/temp/sbatch.printTestTriuGraphDecoder"
    write_to_file("Loaded in data.", filepath=write_fpath)

    # initialize model on device
    #device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = "cpu"

    # TriuGraphTransformer
    model = TriuGraphTransformer(dim_model=102, 
                                encoder_depth=6, 
                                nhead=6, 
                                encoder_mlp_dim=102,
                                decoder_input_dim=102, 
                                decoder_dim_feedforward=102,
                                decoder_depth=6,
                                dim_encoder_head=17, 
                                num_out_nodes=100, 
                                latent_length=102,
                                dropout=0.1)

    model.load_state_dict(torch.load("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainedModels/TriuGraphModel_800.pt"))
    model.eval()
    model.to(device)

    # Find number of parameters
    model_params = sum(p.numel() for p in model.parameters())
    write_to_file(f"Model params: {model_params}", filepath=write_fpath)


    train_targets = []
    test_targets = []

    #train_preds_ith = []
    #test_preds_ith = []

    #train_preds_TOith = []
    #test_preds_TOith = []

    train_preds_decoder = []
    test_preds_decoder = []

    train_maes = []
    test_maes = []

    with torch.no_grad():
        for i, data in enumerate(train_loader):
            inputs, targets, in_targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            decoder_pred, out = triu_graph_greedy_decode(model=model, source=inputs, latent_length=model.input_dim, device=device, b=1, target=in_targets)

            #write_to_file(targets.squeeze(), write_fpath)
            #write_to_file(decoder_pred.squeeze(), write_fpath)


            train_ground_truth[i, :] = targets.squeeze().numpy()
            #train_pred[i, :] = decoder_pred.squeeze().detach().numpy()[2:, 2:][np.triu_indices(100, k=1)]
            train_pred[i, :] = out.detach().numpy()

            train_targets.append(targets)
            train_preds_decoder.append(decoder_pred)

            #write_to_file("MAE TRAIN Target vs TRAIN Prediction ConvDecoder:", filepath=write_fpath)
            #write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])), filepath=write_fpath)
            mae = np.mean(np.abs(train_ground_truth[i, :] - train_pred[i, :]))

            #mae = np.mean(np.abs(targets.squeeze().numpy()[2:, 2:] - decoder_pred.squeeze().detach().numpy()[2:, 2:]))
            train_maes.append(mae)
        

        for i, data in enumerate(test_loader):
            inputs, targets, in_targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0), data[2].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1

            decoder_pred, out = triu_graph_greedy_decode(model=model, source=inputs, latent_length=model.input_dim, device=device, b=1, target=in_targets)

            test_ground_truth[i, :] = targets.squeeze().numpy()
            #test_pred[i, :] = decoder_pred.squeeze().detach().numpy()[2:, 2:][np.triu_indices(100, k=1)]
            test_pred[i, :] = out.detach().numpy()

            test_targets.append(targets)
            test_preds_decoder.append(decoder_pred)

            #write_to_file("MAE TEST Target vs TEST Prediction ConvDecoder:", filepath=write_fpath)
            #write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])), filepath=write_fpath)
            mae = np.mean(np.abs(test_ground_truth[i, :] - test_pred[i, :]))

            test_maes.append(mae)
            
            
            #mae = np.mean(np.abs(targets.squeeze().numpy()[2:, 2:] - decoder_pred.squeeze().detach().numpy()[2:, 2:]))


    write_to_file("TRAIN MAEs:", filepath=write_fpath)
    write_to_file(train_maes, filepath=write_fpath)
    write_to_file("TEST MAEs:", filepath=write_fpath)
    write_to_file(test_maes, filepath=write_fpath)

    np.save("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/TriuGraphModel/train_ground_truth.npy", train_ground_truth)
    np.save("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/TriuGraphModel/train_pred.npy", train_pred)
    np.save("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/TriuGraphModel/test_ground_truth.npy", test_ground_truth)
    np.save("/home/ahmadf/NeuroTranslate/code/SurfToNetmat/TransformerTest/_FyzTests/TrainOut/TriuGraphModel/test_pred.npy", test_pred)






    '''
    for i, data in enumerate(train_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze().unsqueeze(0) #only use unsqueeze(0) if batch size is 1


        predictions = greedy_decode(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)
        convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)

        train_targets.append(targets)
        train_preds_ith.append(predictions[0])
        train_preds_TOith.append(predictions[1])
        train_preds_ConvDecoder.append(convDecoder_pred)

        write_to_file("MAE TRAIN Target vs TRAIN Prediction ith:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - predictions[0].detach().numpy()[pad:])), filepath=write_fpath)

        write_to_file("MAE TRAIN Target vs TRAIN Prediction TOith:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - predictions[1].detach().numpy()[pad:])), filepath=write_fpath)

        write_to_file("MAE TRAIN Target vs TRAIN Prediction ConvDecoder:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])), filepath=write_fpath)

    for i, data in enumerate(test_loader):
        inputs, targets = data[0].to(device), data[1].to(device).squeeze()#.unsqueeze(0) #only use unsqueeze(0) if batch size is 1

        print(inputs.shape)

        predictions = greedy_decode(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)
        convDecoder_pred = greedy_decode_conv(model=model, source=inputs, input_dim=model.input_dim, device=device, b=batch_size)

        test_targets.append(targets)
        test_preds_ith.append(predictions[0])
        test_preds_TOith.append(predictions[1])
        test_preds_ConvDecoder.append(convDecoder_pred)

        write_to_file("MAE TEST Target vs TEST Prediction ith:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - predictions[0].detach().numpy()[pad:])), filepath=write_fpath)

        write_to_file("MAE TEST Target vs TEST Prediction TOith:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - predictions[1].detach().numpy()[pad:])), filepath=write_fpath)

        write_to_file("MAE TEST Target vs TEST Prediction ConvDecoder:", filepath=write_fpath)
        write_to_file(np.mean(np.abs(targets.squeeze().numpy()[pad:] - convDecoder_pred.squeeze().detach().numpy()[pad:])), filepath=write_fpath)
    '''

    '''
    torch.set_printoptions(threshold=10_000)
    write_to_file("Train Target:", filepath=write_fpath)
    write_to_file(train_targets, filepath=write_fpath)

    write_to_file("Test Target:", filepath=write_fpath)
    write_to_file(test_targets, filepath=write_fpath)

    #write_to_file("Train Prediction ith:", filepath=write_fpath)
    #write_to_file(train_preds_ith, filepath=write_fpath)

    #write_to_file("Test Prediction ith:", filepath=write_fpath)
    #write_to_file(test_preds_ith, filepath=write_fpath)

    #write_to_file("Train Prediction TOith:", filepath=write_fpath)
    #write_to_file(train_preds_TOith, filepath=write_fpath)

    #write_to_file("Test Prediction TOith:", filepath=write_fpath)
    #write_to_file(test_preds_TOith, filepath=write_fpath)

    write_to_file("Train Prediction ConvDecoder:", filepath=write_fpath)
    write_to_file(train_preds_ConvDecoder, filepath=write_fpath)

    write_to_file("Test Prediction ConvDecoder:", filepath=write_fpath)
    write_to_file(test_preds_ConvDecoder, filepath=write_fpath)
    '''
    