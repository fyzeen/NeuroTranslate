#!/bin/bash
#SBATCH -J ForwardPass
#SBATCH -o /home/ahmadf/batch/sbatch.out%j
#SBATCH -e /home/ahmadf/batch/sbatch.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 20G # 10G for base, 6.5G para small
#SBATCH --cpus-per-task 10
#SBATCH -t 0-24:00:00  # might depend on epoch, approx 50epoch = 24 hours

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

python3 count_num_params.py

conda activate # not specified means back to (base)