# importing modules
import os
import os.path as op

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import hcp_utils as hcp

import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

def ListNIFTIs(path, type):
        '''
        Lists all the files apropriate .nii files in a directory provided by 'path'
        '''
        if type == 'profumo':
            for f in os.listdir(path):
                if not f.startswith('.'):
                    if f.endswith('.nii') and f.startswith("sub"):
                        yield f
        elif type == "ICA":
            for f in os.listdir(path):
                if not f.startswith('.'):
                    if f.endswith('.nii'):
                        yield f
        elif type == "gradients":
            for f in os.listdir(path):
                if not f.startswith('.'):
                    if f.endswith('_aligned.dtseries.nii') and f.startswith("emb_subj-"):
                        yield f
        else:
            raise ValueError("Type MUST be 'profumo', 'ICA', or 'gradients'")


def ListDirNoHidden(path):
        '''
        Lists all the files non-hidden files in a directory provided by 'path'
        '''
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f


def extract_cortexData(data=None, numFeatures=None):
    '''
    This function extracts all the cortex data from a CIFTI file for all features/dimensions.
    
    Inputs
    ==========
    data: numpy.ndarray or numpy.memmap
        Data stored at each vertex in the CIFTI format, with a shape [numFeatures, numVertices]
        
    numFeatures: int
        The number of features/dimensions stored at each vertex
        
    Ouputs
    ==========
    cortex: numpy.ndarray
        Data stored in each vertex in the CORTEX, with shape [59412 (num cortical vertices with fMRI data), numFeatures]
    
    '''
    rows = []
    for i in range(numFeatures):
        row = hcp.cortex_data(data[i])
        rows.append(row)
    
    cortex = np.stack(rows, axis=1)
    cortex = cortex[~np.all(cortex == 0, axis=1)] # Excluding all vertices without fMRI data
    
    return cortex

def read_Subj(x=None, y=None, numFeatures_x=None, numFeatures_y=None, brain_shape=None):
    '''
    This function reads and formats a subject's data into a clean pytorch_geometric.data.Data object for use in
    translation models.
    
    Inputs
    ==========
    x: str
        Path to the raw CIFTI file (.nii) with the brain rep. data FROM which you want to translate for ONE SUBJECT
        
    y: str
        Path to the raw CIFTI file (.nii) with the brain rep. data TO which you want to translate for ONE SUBJECT
        
    numFeatures_x: int
        Number of dimensions/features in the brain rep. data FROM which you want to translate
    
    numFeatures_x: int
        Number of dimensions/features in the brain rep. data TO which you want to translate
        
    brain_shape: str, ["inflated", "pial", "very_inflated", "flat" or "sphere"]
    
        
    Outputs
    ==========
    out: pytorch_geometric.data.Data object
        PyG Data object with attributes x, y, pos, edge_index, and edge_attr for one subject for one translation
    
    '''
    img_x = nib.load(x)
    img_y = nib.load(y)
    
    coordinatesR = hcp.mesh["inflated_right"][0] # Coordinates (pos) of vertices in HCP S1200 group average meshes per https://github.com/rmldj/hcp-utils/blob/master/hcp_utils/hcp_utils.py
    coordinatesL = hcp.mesh["inflated_left"][0]
    coordinatesR = coordinatesR[hcp.vertex_info.grayr] # Subset of coordinates that represent vertices with fMRI data in HCP
    coordinatesL = coordinatesL[hcp.vertex_info.grayl]
    coordinates = np.vstack((coordinatesL, coordinatesR)) # Stacking coordinates

    data_x = img_x.get_fdata()
    data_y = img_y.get_fdata()
    
    cortex_x = torch.tensor(extract_cortexData(data_x, numFeatures_x)).float() # potentially will have to cast these to float or double...
    cortex_y = torch.tensor(extract_cortexData(data_y, numFeatures_y)).float()
    
    csr_matrix = hcp.cortical_adjacency
    coo_matrix = csr_matrix.tocoo()
    indices = torch.from_numpy(np.vstack((coo_matrix.row, coo_matrix.col)).astype(int))
    
    out = Data(x=cortex_x, y=cortex_y, edge_index=indices, pos=torch.tensor(coordinates))
    
    transform = T.Compose([T.Cartesian()]) # this may need to change based on the convolution used
    out = transform(out)

    return out


def findRawPath(root, type, numFeatures):
    if type == "profumo":
        path = op.join(root, "profumo_reps", f"HCP_Profumo_cifti_bigdata_1200subs_{numFeatures}modes_smooth_5mm_4runs.ppp", "Maps")

        if not op.exists(path):
            raise ValueError("Path does not exist!")
    elif type == "ICA":
        path = op.join(root, "ICA_reps", f"3T_HCP1200_MSMAll_d{numFeatures}_ts2_Z", "component_maps")

        if not op.exists(path):
            raise ValueError("Path does not exist!")
    elif type == "gradients":
        path = op.join(root, "gradient_reps", f"gradmaps_d{numFeatures}")

        if not op.exists(path):
            raise ValueError("Path does not exist!")
    else:
        raise ValueError("Type MUST be 'profumo', 'ICA', or 'gradients'")
    return path

class TranslationsData(Dataset):
    def __init__(self, root="/scratch/ahmadf/NeuroTranslate/", raw_root="/scratch/tyoeasley/brain_representations/",
                x_type=None, y_type=None, numFeatures_x=None, numFeatures_y=None, brain_shape="inflated",
                transform=None, pre_transform=None, pre_filter=None):
        
        self.root = root
        self.raw_root = raw_root
        
        self.x_type = x_type # 'profumo', 'ICA' or 'gradients'
        self.y_type = y_type # 'profumo', 'ICA' or 'gradients'
        self.numFeatures_x = numFeatures_x #int
        self.numFeatures_y = numFeatures_y #int
        self.brain_shape = brain_shape
        
        self.x_raw_path = findRawPath(raw_root, x_type, numFeatures_x)
        self.y_raw_path = findRawPath(raw_root, y_type, numFeatures_y)
                        
        self.subj_list = list(np.genfromtxt(op.join(root, "subj_list.csv"), delimiter=",", dtype=int))
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return {'x':sorted(list(ListNIFTIs(self.x_raw_path, self.x_type))), 'y':sorted(list(ListNIFTIs(self.y_raw_path, self.y_type)))}
    
    @property
    def processed_file_names(self):
        return sorted(list(ListDirNoHidden(self.processed_dir)))
    
    @property
    def processed_dir(self):
        return op.join(self.root, "processed", f"{self.x_type}_d{self.numFeatures_x}_TO_{self.y_type}_d{self.numFeatures_y}")
    

    def process(self):
        for i, x_fname in enumerate(self.raw_file_names['x']):
            y_fname = self.raw_file_names['y'][i]
            
            x_path = op.join(self.x_raw_path, x_fname)
            y_path = op.join(self.y_raw_path, y_fname)
            
            subjData = read_Subj(x=x_path, y=y_path, 
                                 numFeatures_x=self.numFeatures_x, numFeatures_y=self.numFeatures_y, 
                                 brain_shape=self.brain_shape)
                                 
            torch.save(subjData, op.join(self.processed_dir, f"subj{self.subj_list[i]}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if idx<1003:
            subj_id = self.subj_list[idx]
        else:
            subj_id = idx
        return torch.load(op.join(self.processed_dir, f"subj{subj_id}.pt"))
    

class LOCALTranslationsData(Dataset):
    def __init__(self, root, x_type=None, y_type=None, numFeatures_x=None, numFeatures_y=None, brain_shape="inflated",
                 transform=None, pre_transform=None, pre_filter=None):
        
        self.root = root
        
        self.x_type = x_type
        self.y_type = y_type
        self.numFeatures_x = numFeatures_x
        self.numFeatures_y = numFeatures_y
        self.brain_shape = brain_shape
        
        self.x_raw_path = op.join(root, "raw", f"{x_type}_d{numFeatures_x}")
        self.y_raw_path = op.join(root, "raw", f"{y_type}_d{numFeatures_y}")
                        
        self.subj_list = list(np.genfromtxt(op.join(root, "LOCAL_subj_list.csv"), delimiter=",", dtype=int))
        
        super().__init__(root, transform, pre_transform, pre_filter)
        
    @property
    def raw_file_names(self):
        return {'x':sorted(list(ListDirNoHidden(self.x_raw_path))), 'y':sorted(list(ListDirNoHidden(self.y_raw_path)))}
    
    @property
    def processed_file_names(self):
        return sorted(list(ListDirNoHidden(self.processed_dir)))
    
    @property
    def processed_dir(self):
        return op.join(self.root, "processed", f"{self.x_type}_d{self.numFeatures_x}_TO_{self.y_type}_d{self.numFeatures_y}")
    

    def process(self):
        for i, x_fname in enumerate(self.raw_file_names['x']):
            y_fname = self.raw_file_names['y'][i]
            
            x_path = op.join(self.x_raw_path, x_fname)
            y_path = op.join(self.y_raw_path, y_fname)
            
            subjData = read_Subj(x=x_path, y=y_path, 
                                 numFeatures_x=self.numFeatures_x, numFeatures_y=self.numFeatures_y, 
                                 brain_shape=self.brain_shape)
                                 
            torch.save(subjData, op.join(self.processed_dir, f"subj{self.subj_list[i]}.pt"))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(op.join(self.processed_dir, f"subj{self.subj_list[idx]}.pt"))