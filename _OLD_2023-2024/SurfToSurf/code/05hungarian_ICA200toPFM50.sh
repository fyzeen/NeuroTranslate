#!/bin/bash
#SBATCH --job-name=TestConvTrain
#SBATCH --output=/home/ahmadf/batch/sbatch.out%j
#SBATCH --error=/home/ahmadf/batch/sbatch.err%j
#SBATCH --time=23:55:00
#SBATCH --mem=64GB
#SBATCH -N 1 #number nodes
#SBATCH -n 1 #number tasks per node


module load python
source activate neurotranslate

python3 /home/ahmadf/NeuroTranslate/code/05hungarian_ICA200toPFM50.py

conda deactivate

