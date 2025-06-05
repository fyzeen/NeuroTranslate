#!/bin/bash
#SBATCH -J "ts2netmats"
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ts2netmats.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/ts2netmats.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 10G 
#SBATCH --cpus-per-task 10
#SBATCH -t 0-24:00:00 

module load fsl

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/scripts/ts2netmats.py"

dir_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/timeseries/transposed_ts/"
Fnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/netmats"
Pnetmats_output="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/partial_netmats"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/ABCD_ICA_subjids.txt"

fslipython ${script_path} ${dir_path} ${Fnetmats_output} ${Pnetmats_output} ${subjects_list_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/
