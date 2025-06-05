import numpy as np
import nibabel as nib
from nibabel.cifti2 import BrainModelAxis, SeriesAxis, Cifti2Image

print(nib.__version__)

# Load the CIFTI file
orig_img = nib.load('/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/DualRegression_ABCD_CIFTI/local_data/melodic_IC.nii')
orig_header = orig_img.header
orig_axes = orig_img.header.matrix

# Extract the brain model axis (axis 1)
brain_axis = BrainModelAxis.from_index_mapping(orig_img.header.get_index_map(1))

# Create a new time axis with 100 time points
time_axis = SeriesAxis(start=0.0, step=1.0, size=100)  # For dtseries.nii

# Generate new data with shape (100, 91282)
new_data = np.random.rand(100, 91282).astype(np.float32)

# Create the new CIFTI2 image from axes
new_img = Cifti2Image(new_data, (time_axis, brain_axis))

# Save to a new file
nib.save(new_img, "./local_data/new_100x91282.dtseries.nii")

# cifti_img = nib.load('/Users/fyzeen/FyzeenLocal/GitHub/NeuroTranslate/DualRegression_ABCD_CIFTI/local_data/dr_out/map.nii')
# # Get the data
# data = cifti_img.get_fdata()  # shape: (timepoints or scalars, brain models)

# # Get the header and brain model
# header = cifti_img.header

# # Print some information
# print(data)