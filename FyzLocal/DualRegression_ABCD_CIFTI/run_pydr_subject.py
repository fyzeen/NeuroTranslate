import sys
sys.path.append('/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/dr_cifti_scripts/pydr-main') 
import pydr.dr as pydr

import argparse

# Create the parser and add arguments
parser = argparse.ArgumentParser()

parser.add_argument("-func", required=True, help="Path to the subject's functional CIFTI fMRI time series file (e.g., .dtseries.nii). This is the main input data.")
parser.add_argument("-map", required=True, help="Output path for the subject-specific spatial maps (CIFTI format), generated in dual regression stage 2. (should be something like map.nii)")
parser.add_argument("-timecourse", required=True, help="Output path for the extracted time series (1st DR stage) after regressing the group spatial maps onto the subject's data. (should be something like timeseries.csv)")
parser.add_argument("-grp_map", required=True, help="Path to the group-level spatial maps in CIFTI format, used in dual regression stage 1.")

parser.add_argument("--amplitude", default=None, help="Output path for the amplitude of the extracted time series. Default: None")
parser.add_argument("--spectra", default=None, help="Output path for the power spectra of the extracted timecourses. Default: None")
parser.add_argument("--func_smooth", default=None, help="Path to a smoothed version of the func file in CIFTI format. Default: None")

args = parser.parse_args()

# Run pydr for single subject
pydr.dr_single_subject(func = args.func,
                       map = args.map,
                       timecourse = args.timecourse,
                       grp_map = args.grp_map,
                       amplitude = args.amplitude,
                       spectra = args.spectra,
                       func_smooth = args.func_smooth)


