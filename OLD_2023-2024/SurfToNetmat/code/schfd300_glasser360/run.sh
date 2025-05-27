#!/bin/bash
#SBATCH -J ICAd15_schfd100
#SBATCH -o /home/ahmadf/batch/temp1/sbatch.out%j
#SBATCH -e /home/ahmadf/batch/temp1/sbatch.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 10G 
#SBATCH --cpus-per-task 10
#SBATCH -t 0-72:00:00 

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

#python3 traintest_PCAConvTransformer.py
#python3 traintest_PCAVanillaKrakencoder.py
#python3 traintest_PCAkrakloss_encoder.py
#python3 traintest_PCAVariationalkrakloss_encoder.py

python3 test_PCAConvTransformer.py
python3 test_PCAkrakloss_encoder.py
#python3 test_PCAVanillaKrakencoder.py
python3 test_PCAVariationalkrakloss_encoder.py

conda activate # not specified means back to (base)