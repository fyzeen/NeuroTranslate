#!/bin/bash
#SBATCH -J "TransposeABCD_tseries"
#SBATCH -o /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/TransposeABCD_tseries.out%j
#SBATCH -e /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/logs/TransposeABCD_tseries.err%j
#SBATCH --partition=tier2_cpu 
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 4G 
#SBATCH --cpus-per-task 10
#SBATCH -t 0-23:00:00 

source activate neurotranslate

script_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/scripts/transpose_ptseries.py"
subjects_list_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/generate_ICA/ABCD_ICA_subjids.txt"
untranspose_ts_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/timeseries/"
save_path="/ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/schaefer100/timeseries/transposed_ts/"

python ${script_path} ${subjects_list_path} ${untranspose_ts_path} ${save_path}

chmod -R 771 /ceph/chpc/shared/janine_bijsterbosch_group/naranjorincon_scratch/NeuroTranslate/ABCD_NetMats/