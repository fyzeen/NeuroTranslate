import numpy as np
import pandas as pd
import nibabel as nib

def matrix_to_mesh(input_mat, num_channels, tri_indices_ico6subico2_fpath, out_fpath=None):
    '''
    This function will take a numpy array of size [num_channels, 320, 153] and transform it into a shape.gii (GIFTI) file to overlay on an ico6 surface.

    Inputs 
    ----------
    input_mat: np.ndarray
        Array of shape [num_channels, 320, 153] containing the surface information for a single subject you want to make into a shape.gii file

    num_channels: int
        Number of channels in the surface (e.g., number if ICA dims)

    tri_indices_ico6subico2_fpath: str
        Path to file with mapping from 320x153 matrix to ico6 sphere vertices. This csv is also available here: https://github.com/metrics-lab/surface-vision-transformers/blob/main/utils/triangle_indices_ico_6_sub_ico_2.csv

    out_fpath: str 
        Path to which you'd like to save the generated GIFTI file. Default: None will NOT save the GIFTI file

    Outputs
    ----------
    out: nib.GiftiImage
        GIFTI image filled in with surface information stored in the input matrix
    '''

    indices_mesh_triangles = pd.read_csv(tri_indices_ico6subico2_fpath)
    mesh_vec = np.zeros([num_channels, 40962])

    for i in range(num_channels):
        for j in range(320):
            indices_to_insert = indices_mesh_triangles[str(j)].to_numpy()
            mesh_vec[i, indices_to_insert] = input_mat[i, j, :]

    out = nib.GiftiImage()
    for i in range(num_channels):
        out.add_gifti_data_array(nib.gifti.GiftiDataArray(mesh_vec[i, :].astype("float32")))

    if out_fpath is not None:
        out.to_filename(out_fpath)

    return out
