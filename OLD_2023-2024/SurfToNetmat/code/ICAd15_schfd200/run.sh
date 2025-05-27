#!/bin/bash
#SBATCH -J ICAd15_schfd200
#SBATCH -o /home/ahmadf/batch/sbatch.out%j
#SBATCH -e /home/ahmadf/batch/sbatch.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 10G 
#SBATCH --cpus-per-task 10
#SBATCH -t 0-24:00:00 

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

python3 train_ConvTransformer.py
#python3 test_ConvTransformer.py

#python3 train_GraphTransformer.py
#python3 test_GraphTransformer.py

#python3 train_TriuGraphTransformer.py
#python3 test_TriuGraphTransformer.py


conda activate # not specified means back to (base)