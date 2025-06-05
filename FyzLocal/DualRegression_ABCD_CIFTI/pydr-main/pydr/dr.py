#!/usr/bin/env python
import os
import sys
import math
import shlex
import nibabel as nb
import numpy as np
import os.path as op
from tempfile import TemporaryDirectory as tempdir
from subprocess import check_output as run
import pandas as pd
from typing import Optional
import scipy.signal as sig
import logging

LOG_FORMAT = "[%(asctime)s - %(name)s.%(funcName)s ] %(levelname)s : %(message)s"
LOG_LEVEL = logging.DEBUG

DR_PATH = op.dirname(op.abspath(__file__))


def assert_env(env):
    if os.getenv(env) is None:
        raise RuntimeError(f'Environment variable {env} is not set.')


def fold(d):

    n_t = d.shape[1]
    n_v = d.shape[0]

    dim = np.ceil(n_v / 100.0).astype(int)

    d0 = np.zeros((100 * dim, n_t))
    d0[:n_v, :] = d
    d0 = d0.reshape((10, 10, dim, n_t))

    mask = np.ones((100 * dim))
    mask[n_v:] = 0
    mask = mask.reshape((10, 10, dim))

    return d0, mask.astype(np.int16)


def basename(f):
    return op.basename(op.splitext(op.splitext(f)[0])[0])


def convert_cifti_to_dummy(cifti_name, nifti_name):
    d0, m0 = fold(nb.load(cifti_name).get_fdata(caching="unchanged").T)

    dummy = nifti_name + ".dummy.nii.gz"
    dummy_mask = nifti_name + "_mask.dummy.nii.gz"

    nb.Nifti1Image(d0, affine=np.eye(4)).to_filename(dummy)
    nb.Nifti1Image(m0, affine=np.eye(4)).to_filename(dummy_mask)

    return dummy, dummy_mask


def convert_dummy_to_cifti(nifti, mask, ref_cifti, outname, dtype="dtseries"):

    dd = nb.load(nifti)
    d0 = dd.get_fdata()[nb.load(mask).get_fdata().astype(bool)]

    cifti = nb.load(ref_cifti)

    bm_full = cifti.header.get_axis(1)

    if dtype == "dtseries":
        series = nb.cifti2.SeriesAxis(start=0, step=1, size=d0.shape[-1])
    elif dtype == "dscalar":
        series = nb.cifti2.ScalarAxis(name=[f"{i}" for i in np.arange(d0.shape[-1])])
    else:
        raise RuntimeError(f"Unknown dtype: {dtype}")

    header = nb.cifti2.Cifti2Header.from_axes((series, bm_full))
    nb.Cifti2Image(d0.T, header=header).to_filename(outname)


def fsl_glm(func, design, output, demean=False, mask=None, out_z=None, des_norm=False):

    cmd = [
        f"{os.getenv('FSLDIR')}/bin/fsl_glm",
        "-i",
        func,
        "-d",
        design,
        "-o",
        output,
    ]

    if demean:
        cmd.append("--demean")
    if mask is not None:
        cmd.append("-m")
        cmd.append(mask)
    if out_z is not None:
        cmd.append(f"--out_z={out_z}")
    if des_norm:
        cmd.append("--des_norm")

    return cmd


def nanmad(d, axis=None, keepdims=False):
    """
    Calculate the median absolute deviation (MAD) accounting for NaNs.
    (see: https://en.wikipedia.org/wiki/Median_absolute_deviation)
    """
    median = np.nanmedian(d, axis=axis, keepdims=keepdims)
    diff = d - median
    mad = np.nanmedian(np.abs(diff), axis=axis, keepdims=keepdims)

    return mad


def calc_spec(timeseries, TR=0.392, nfft=1024):
    """
    Calculate spectra with Welch's method.
    """

    fs = 1 / TR

    ext = op.splitext(timeseries)[1]

    if ext in [".csv"]:
        d = np.loadtxt(timeseries, delimiter=",")
    elif ext in [".txt"]:
        d = np.loadtxt(timeseries)
    else:
        raise RuntimeError("doh")

    d = (d - np.mean(d, axis=0)) / np.std(d, axis=0)

    Pxx = np.zeros((int(np.round((nfft / 2) + 1)), d.shape[-1]))

    for idx, m in enumerate(d.T):
        f, Pxx[:, idx] = sig.welch(m, fs=fs, nperseg=nfft)

    return np.concatenate((f[:, np.newaxis], Pxx), axis=1)


def dr_single_subject(
    func:              str,
    map:               str,
    timecourse:        str,
    grp_map:           str,
    amplitude:         str = None,
    spectra:           str = None,
    func_smooth:       str = None
):

    assert_env('FSLDIR')
    assert_env('FSLOUTPUTTYPE')

    if not op.exists(op.dirname(map)):
        os.makedirs(op.dirname(map))

    if not op.exists(op.dirname(timecourse)):
        os.makedirs(op.dirname(timecourse))

    # -- convert group_map cifti to nifti --
    td = tempdir(dir=op.dirname(map))
    tmp_grp_map, tmp_grp_mask = convert_cifti_to_dummy(grp_map, f"{td.name}/grp_maps")


    # --- convert cifti func to nifti ---
    tr = nb.load(func).header.get_axis(0).step
    n_time = nb.load(func).header.get_axis(0).size
    func_nifti = convert_cifti_to_dummy(func, f"{td.name}/func")[0]


    # --- convert smoothed cifti func to nifti ---
    if func_smooth:
        tr = nb.load(func_smooth).header.get_axis(0).step
        n_time = nb.load(func_smooth).header.get_axis(0).size
        func_smoothed_nifti = convert_cifti_to_dummy(func_smooth, f"{td.name}/func_smooth")[0]


    # If there is no smoothed nifti, use the same input for dualreg1 and dualreg2
    else:
        func_smoothed_nifti = func_nifti


    # --- DR stage 1: regress spatial maps onto functional timeseries ---

    cmd = fsl_glm(
        func=func_nifti, design=tmp_grp_map, output=timecourse, demean=True, mask=tmp_grp_mask
    )
    run(cmd)

    # DR stage 2: regress stage 1 timeseries onto functional timeseries

    tmp_map = f"{td.name}/func_stage2.nii.gz"
    tmp_map_z = f"{td.name}/func_stage2_z.nii.gz"

    cmd = fsl_glm(
        func=func_smoothed_nifti,
        design=timecourse,
        output=tmp_map,
        demean=True,
        mask=tmp_grp_mask,
        des_norm=True,
        out_z=tmp_map_z,
    )
    run(cmd)

    convert_dummy_to_cifti(
        nifti=tmp_map,
        mask=tmp_grp_mask,
        ref_cifti=grp_map,
        outname=map,
        dtype="dscalar",
    )

    convert_dummy_to_cifti(
        nifti=tmp_map_z,
        mask=tmp_grp_mask,
        ref_cifti=grp_map,
        outname=map.replace(".dscalar.nii", "_Z.dscalar.nii"),
        dtype="dscalar",
    )

    # DR Amplitude

    if amplitude is not None:

        d = np.loadtxt(timecourse)
        d = nanmad(d, axis=0)

        if not op.exists(op.dirname(amplitude)):
            os.makedirs(op.dirname(amplitude))

        np.savetxt(amplitude, d, delimiter=",")

    # DR Spectra

    if spectra is not None:

        spec = calc_spec(timecourse, nfft=n_time, TR=tr)

        if not op.exists(op.dirname(spectra)):
            os.makedirs(op.dirname(spectra))

        np.savetxt(spectra, spec, delimiter='\t', fmt='%.6f')


def dr(files, grp_maps, workdir, submit_to_cluster=True):

    assert_env('FSLDIR')
    assert_env('FSLOUTPUTTYPE')

    while op.exists(workdir):
        workdir += "+"
    if not op.exists(workdir):
        os.makedirs(workdir)

    #  setup logging

    logdir = f'{workdir}/logs'
    if not op.exists(logdir):
        os.makedirs(logdir)

    logging.basicConfig(
        filename=f"{logdir}/pydr.log", format=LOG_FORMAT, level=LOG_LEVEL
    )

    logger = logging.getLogger(__name__)
    logger.info(f"pyDR (pid={os.getpid()})")

    logger.info(f'Input files: {files}')
    logger.info(f"Group mask: {grp_maps}")
    logger.info(f"Working directory: {workdir}")

    # generate DR jobs

    PYTHON_BIN = sys.executable

    with open(f'{workdir}/logs/dr_commands.txt', 'w') as fpr:
        df = pd.read_csv(files)

        for elements in df.itertuples():

            # The csv file can have both an optional smoothed input
            fname_smooth = None
            if len(elements) == 4:
                (idx, subid, fname, fname_smooth) = elements
            elif len(elements) == 3:
                (idx, subid, fname) = elements
            else:
                print("The csv file " + files + " does not have the correct format.")
                exit(1)

            args = [
                fname,
                f"{workdir}/maps/{subid}.dscalar.nii",
                f"{workdir}/timecourses/{subid}.txt",
                grp_maps,
                f"{workdir}/amplitude/{subid}.txt",
                f"{workdir}/spectra/{subid}.txt",
            ]

            if fname_smooth:
                args.append(fname_smooth)

            cmd = f"{PYTHON_BIN} {DR_PATH}/dr_helper.py {' '.join(args)}"

            logger.info(cmd)
            fpr.write(cmd + '\n')

    # Run all commands sequentially in case user does not want to run them in the cluster 
    if not submit_to_cluster:
        with open(f'{workdir}/logs/dr_commands.txt', 'r') as fpr:
            for command in fpr.readlines():
                logger.info(command)
                output=run(shlex.split(command))
                print(output)

    # Run all commands in the cluster with fsl_sub
    else:
        command =[
            'fsl_sub',
            '-T', '1400',
            '-l', f'{workdir}/logs',
            '--export', f'FSLDIR={os.environ["FSLDIR"]}',
            '--export', f'FSLOUTPUTTYPE={os.environ["FSLOUTPUTTYPE"]}',
            '-t', f'{workdir}/logs/dr_commands.txt',
         ]

        logger.info(' '.join(command))
        run(command)
