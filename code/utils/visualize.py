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

def plotHCPSurface(data, surf=hcp.mesh.inflated, bg_map=hcp.mesh.sulc, threshold=0):
    '''
    This function uses nilearn.plootting.view_surf to visualize a surface data on standard HCP surfaces
    '''
    plot = plotting.view_surf(surf, data, threshold=threshold, bg_map=bg_map)
    plot.open_in_browser()
