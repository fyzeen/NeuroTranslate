import pandas as pd
import os
import sys

subject_list_path=sys.argv[1]
dir_path=sys.argv[2]
save_path=sys.argv[3]

# get the subject ids
with open(subject_list_path, 'r') as file:
    subject_ids = file.read().splitlines()


ts_dir = save_path

for subject_id in subject_ids:
    file_path = os.path.join(dir_path, f'untranspose_{subject_id}.txt')
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, sep='\t',index_col=None, header=None)
        df_transposed = df.T
        transposed_file_path = os.path.join(ts_dir, f'{subject_id}.txt')
        df_transposed.to_csv(transposed_file_path, sep='\t', index=False,header=False)
        print(f'Transposed file saved as: {transposed_file_path}')
    else:
        print(f"File for subject ID {subject_id} not found.")
