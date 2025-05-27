#!/bin/bash
#SBATCH -J ICAd15_schfd100
#SBATCH -o /home/ahmadf/batch/sbatch.out%j
#SBATCH -e /home/ahmadf/batch/sbatch.err%j
#SBATCH --partition=tier2_cpu
#SBATCH --account=janine_bijsterbosch
#SBATCH --mem-per-cpu 20G 
#SBATCH --cpus-per-task 1
#SBATCH -t 0-1:00:00 

## IF YOU TRAIN ANYTHING: CHANGE WALLTIME TO 72 HRS, mem-per-cpu to 10G, and cpus-per-task to 10

source activate neurotranslate
echo Activated environment with name: $CONDA_DEFAULT_ENV

#python3 train_ConvTransformer.py
#python3 test_ConvTransformer.py

#python3 train_GraphTransformer.py
#python3 test_GraphTransformer.py

#python3 train_TriuGraphTransformer.py
#python3 test_TriuGraphTransformer.py

#python3 test_krakloss_ConvTransformer.py
#python3 test_PCAkrakloss_encoder.py

#python3 test_PCAVariationalConvTransformer.py

#python3 traintest_ConvTransformer.py
#python3 traintest_krakloss_encoder.py
#python3 traintest_krakloss_ConvTransformer.py
#python3 traintest_VariationalConvTransformer.py

#python3 traintest_PCAConvTransformer.py
#python3 traintest_PCAVariationalConvTransformer.py
#python3 traintest_PCAVariationalKrakLossConvTransformer.py
#python3 traintest_PCAkrakloss_encoder.py
#python3 traintest_PCAVariationalkrakloss_encoder.py
#python3 traintest_PCAVanillaKrakencoder.py

#python3 traintest_PCATwoHemi_krakloss_encoder.py

#python3 meshPCAtest.py

#python3 non_demeaned_corrs_allmodels.py
#python3 test_PCATwoHemi_krakloss_encoder.py
#python3 test_PCAVanillaKrakencoder.py


python3 regression_translations.py

conda activate # not specified means back to (base)