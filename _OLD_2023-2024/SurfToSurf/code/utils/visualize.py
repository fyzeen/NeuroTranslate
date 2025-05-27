# importing modules
import os
import os.path as op

import nibabel as nib
import nilearn.plotting as plotting
import numpy as np
import matplotlib.pyplot as plt
import hcp_utils as hcp

import torch

from .models import * 
from .dataset import *

def cortexToSurfaceVertices(array):
    '''
    This function takes an array (shape [59412, n], where n is the number of features at each node) and transforms it into an array
    with shape [64984, n] such that it can be plotted in an HCP mesh.

    We must use this function because HCP fMRI data are defined on a subset of the surface vertices (29696 out of 32492 for the left 
    cortex and 29716 out of 32492 for the right cortex).

    Input
    ==========
    array: np.ndarray
        Array of shape [59412, n], where n is the number of features at each node

    Output
    ==========
    out: np.ndarray
        Array of shape [64984, n] with the same data, now in a "plottlable" form
    '''
    out = np.zeros((64984, array.shape[1]))
    out[hcp.vertex_info.grayl] = array[:29696]
    out[32492 + hcp.vertex_info.grayr] = array[29696:]
    return out

def plotHCPSurface(data, surf=hcp.mesh.inflated, bg_map=hcp.mesh.sulc, threshold=0, save_path=None):
    '''
    This function uses nilearn.plootting.view_surf to visualize a surface data on standard HCP surfaces
    '''
    plot = plotting.view_surf(surf, data, threshold=threshold, bg_map=bg_map)
    plot.open_in_browser()

    if save_path is not None:
        plot.save_as_html(save_path)


def listStateDictFiles(path, model_type):
    '''
    Helper function for testingForwardPass()
    '''
    for f in os.listdir(path):
        if f.startswith(model_type):
            yield f

def testingForwardPass(dataset, data, model, device, model_type):
    '''
    This function pass your data through a the inputted model (whose stat_dict is loaded from a standardized location on Fyzeen's machine)
    '''
    x_type, numFeatures_x, y_type, numFeatures_y = dataset.x_type, dataset.numFeatures_x, dataset.y_type, dataset.numFeatures_y
    if model_type == "shallowSPLINECONV_EPOCH" or model_type == "SPLINECONV_EPOCH":
        rootpath = f"/scratch/ahmadf/NeuroTranslate/saved_models/{x_type}_d{numFeatures_x}_to_{y_type}_d{numFeatures_y}/"
    else:
        rootpath = f"/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/SurfToSurf/saved_models/{x_type}_d{numFeatures_x}_to_{y_type}_d{numFeatures_y}/"
        #rootpath = f"/scratch/ahmadf/NeuroTranslate/saved_models/{x_type}_d{numFeatures_x}_to_{y_type}_d{numFeatures_y}/" # THIS WAS USED FOR ICC COMPUTATIONS, COMMENT OUT FOR LOCAL VIZUALIZATION!
    file_list = sorted(list(listStateDictFiles(rootpath, model_type)))
    state_dict_path = op.join(rootpath, file_list[-1])

    print(f"Loading from: {state_dict_path}")

    model.load_state_dict(torch.load(state_dict_path, map_location=device))

    model.eval()
    with torch.no_grad():   
        pred = model(data.to(device)).to(device)

    return pred, model
