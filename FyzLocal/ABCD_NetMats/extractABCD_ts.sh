#!/bin/bash
#SBATCH -J "ExtractABCD_Timeseries"
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ExtractABCD_Timeseries.out%j
#SBATCH -e/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ExtractABCD_Timeseries.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 6
#SBATCH -t 0-12:00:00 

module load workbench

file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/ABCD_ICA_subjids.txt"
parcel_file="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/Schaefer2018_100Parcels_17Networks_order.dlabel.nii"
dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/timeseries"

while IFS= read -r subject_id; 
do
    subject_id=$(echo "$subject_id" | tr -d '\r')
    echo "Extracting Timeseries For: $subject_id"

    wb_command -cifti-parcellate \
        /ceph/chpc/rcif_datasets/abcd/abcd_collection3165/derivatives/abcd-hcp-pipeline/sub-${subject_id}/ses-baselineYear1Arm1/files/MNINonLinear/Results/task-rest_DCANBOLDProc_v4.0.0_Atlas.dtseries.nii \
        ${parcel_file} \
        COLUMN \
        ${dir_path}/${subject_id}.ptseries.nii

    wb_command -cifti-convert -to-text \
        ${dir_path}/${subject_id}.ptseries.nii \
        ${dir_path}/untranspose_${subject_id}.txt

done < "${file}"

chmod 771 ${dir_path}/*