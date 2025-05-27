import os
import os.path as op

import numpy as np

import torch
import torch.nn.functional as F

def cosine_distance(tensor1, tensor2):
    # Normalize the tensors along the last dimension (rows) to compute cosine similarity
    tensor1_norm = F.normalize(tensor1, dim=1)
    tensor2_norm = F.normalize(tensor2, dim=1)
    
    # Compute the dot product between every row in the tensors
    dot_product = torch.matmul(tensor1_norm, tensor2_norm.T)
    
    # Calculate the cosine distance using the dot product
    cosine_dist = 1 - dot_product
    
    return cosine_dist

def cosine_distance_columns(tensor1, tensor2):
    # Transpose the tensors to treat columns as rows
    tensor1_transposed = tensor1.T
    tensor2_transposed = tensor2.T
    
    # Compute the cosine distance between columns using the cosine_distance function
    cosine_dist = cosine_distance(tensor1_transposed, tensor2_transposed)
    
    return cosine_dist

def reorder_columns(tensor, out_tensor, mapping):
    new_tensor = torch.empty_like(out_tensor)
    for i, j in mapping:
        new_tensor[:, j] = tensor[:, i]
    return new_tensor