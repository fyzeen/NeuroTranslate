import sys
import os
import math

sys.path.append('../')
sys.path.append('./')
sys.path.append('../../')

from utils.models import *
from utils.utils import *

import numpy as np
from sklearn.decomposition import PCA


if __name__ == "__main__":
    translation = "ICAd15_schfd100"

    # loads in np train meshes
    train_meshes = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    test_meshes = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")

    print("Train Mesh Shape:")
    print(train_meshes.shape)
    print("Test Mesh Shape:")
    print(test_meshes.shape)


    train_mesh_flat = train_meshes.reshape(train_meshes.shape[0], -1) 
    test_mesh_flat = test_meshes.reshape(test_meshes.shape[0], -1) 

    print("Train Flat Mesh Shape:")
    print(train_mesh_flat.shape)
    print("Test Flat Mesh Shape:")
    print(test_mesh_flat.shape)

    pca = PCA(n_components=500)
    pca.fit(train_mesh_flat)

    eigenvalues = pca.explained_variance_
    print("-------------------- EIGEN VALUES --------------------")
    print(eigenvalues)

    # Testing performance
    train_transform = pca.transform(train_mesh_flat)
    train_pred = pca.inverse_transform(train_transform)

    test_transform = pca.transform(test_mesh_flat)
    test_pred = pca.inverse_transform(test_transform)

    train_mean = np.mean(train_mesh_flat, axis=0)

    test_corrs = []
    for i in range(test_mesh_flat.shape[0]):
        corr = np.corrcoef(test_mesh_flat[i, :] - train_mean, test_pred[i, :] - train_mean)
        test_corrs.append(corr[0,1])

    train_corrs = []
    for i in range(train_mesh_flat.shape[0]):
        corr = np.corrcoef(train_mesh_flat[i, :] - train_mean, train_pred[i, :] - train_mean)
        train_corrs.append(corr[0,1])

    print("-------------------- DEMEANED TRAIN CORRs --------------------")
    print(train_corrs)

    print("-------------------- DEMEANED TEST CORRs --------------------")
    print(test_corrs)





