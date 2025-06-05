import os
import sys
import numpy as np
import pandas as pd
from fsl import nets

ts_dir = sys.argv[1]
Fnetmats_output=sys.argv[2]
Pnetmats_output=sys.argv[3]
subject_list_path=sys.argv[4]

# get the subject ids
with open(subject_list_path, 'r') as file:
    subject_ids = file.read().splitlines()

# Calculating netmats for each subject
ts = nets.load(ts_dir, 0.72, varnorm=0)

# full netmats
nmats = np.zeros((ts.ndatasets, ts.nnodes ** 2))

for i, subj, run, data in ts.allts:
    data = data.T
    n = data.shape[0]
    corr = np.corrcoef(data)
    # Set diagonal elements to 0
    #corr[np.diag_indices(n)] = 0
    #corr = 0.5 * np.log((1 + corr) / (1 - corr)) # r to z
    nmats[i] = corr.flatten()
Fnetmats=nmats
n = int(np.sqrt(Fnetmats.shape[1]))
print(n)
square_matrices = np.reshape(Fnetmats, (Fnetmats.shape[0], n, n)) # reshape into square matrix
for i, matrix in enumerate(square_matrices):
    filename=subject_ids[i] + '.csv'
    np.savetxt(os.path.join(Fnetmats_output,filename), matrix, delimiter=',')


# partial netmats
nmats = np.zeros((ts.ndatasets, ts.nnodes ** 2))
rho=0.1
partial='TRUE'
for i, subj, run, data in ts.allts:
    n    = data.shape[1]
    corr = np.cov(data.T)
    # Regularize the covariance matrix
    corr = corr / np.sqrt(np.mean(corr.diagonal(0)**2))
    corr = np.linalg.inv(corr + (rho * np.eye(n)))
    if partial:
        # Compute partial correlations
        corr = -corr
        diags = np.sqrt(np.abs(corr.diagonal(0)))
        diags = np.tile(diags, (1, n)).reshape((n, n))
        corr  = (corr / diags.T) / diags
        #corr[np.diag_indices(n)] = 0
        #corr = 0.5 * np.log((1 + corr) / (1 - corr)) # r to z
        nmats[i] = corr.flatten()
Pnetmats=nmats
n = int(np.sqrt(Pnetmats.shape[1]))
print(n)
square_matrices = np.reshape(Pnetmats, (Pnetmats.shape[0], n, n)) # reshape into square matrix
for i, matrix in enumerate(square_matrices):
    filename=subject_ids[i] + '.csv'
    np.savetxt(os.path.join(Pnetmats_output,filename), matrix, delimiter=',')
