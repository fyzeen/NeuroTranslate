import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def reshape_meshes(meshes):
    return meshes.reshape(meshes.shape[0], -1)

def separate_hemis(arr):
    num_obs = arr.shape[0]
    L_hemi = arr[:int(num_obs/2)]
    R_hemi = arr[int(num_obs/2):]

    return L_hemi, R_hemi

def process_hemi(arr, sub_IDs, id_df):
    df = pd.DataFrame(arr)

    df["index"] = id_df.loc[sub_IDs[0] - 1][3].to_list()
    df = df.set_index("index")

    return df

def process_hemis(arr, sub_IDs, id_df):
    L_hemi, R_hemi = separate_hemis(arr)
    L_hemi_df = process_hemi(L_hemi, sub_IDs, id_df)
    R_hemi_df = process_hemi(R_hemi, sub_IDs, id_df)

    return L_hemi_df, R_hemi_df

def concat_dfs(arr1, arr2):
    return pd.concat([arr1, arr2])


def univariate_regressions(X_df, y_df, pheno="CogTotalComp_Unadj"):
    # Prepare to store results
    results = []

    # Align
    common_indices = X_df.index.intersection(y_df.index)
    X_df = X_df.loc[common_indices]
    y_df = y_df.loc[common_indices]

    # Perform regression for each feature column
    for feature in X_df.columns:
        # Extract the feature column and target variable
        X_feature = X_df[feature]
        y = y_df[pheno]

        X_feature = X_feature.to_numpy().reshape(-1,1)
    
        # Create and fit the model
        model = LinearRegression()
        model.fit(X_feature, y)
        y_pred = model.predict(X_feature)
        
        # Store the results
        results.append({
            'feature': feature,
            'coefficient': float(model.coef_[0]),
            'intercept': float(model.intercept_),
            'r2': r2_score(y, y_pred)
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    translation = "ICAd15_schfd100"

    # Load in subject IDs and phenotypes
    path = "/home/ahmadf/NeuroTranslate/SurfToNetmat/code/ICAd15_schfd100/regression_tests/local_data"

    id_df = pd.read_csv(f"{path}/actual_subID.csv", header=None)
    pheno_df = pd.read_csv(f"{path}/hcp_ya_behv.csv")
    pheno_df = pheno_df.set_index("Subject")
    pheno_df = pheno_df.dropna(subset=["CogTotalComp_Unadj"]) # drop subjects with NaNs in CogTotalComp_Unadj

    test_subIDs = pd.read_csv(f"{path}/actual_test_subID.csv", header=None)
    valid_subIDs = pd.read_csv(f"{path}/actual_valid_subID.csv", header=None)
    train_subIDs = pd.read_csv(f"{path}/actual_train_subID.csv", header=None)


    # load in data
    train_meshes = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_data.npy")
    train_netmats = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/train_labels.npy")

    test_meshes = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_data.npy")
    test_netmats = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/test_labels.npy")

    validation_meshes = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/validation_data.npy")
    validation_netmats = np.load(f"/scratch/naranjorincon/surface-vision-transformers/data/{translation}/template/validation_labels.npy")

    print("Loaded in data.")

    # reshape meshes
    train_meshes = reshape_meshes(train_meshes)
    test_meshes = reshape_meshes(test_meshes)
    validation_meshes = reshape_meshes(validation_meshes)

    # Processing
    train_meshes_L, train_meshes_R = process_hemis(train_meshes, train_subIDs, id_df)
    train_netmats, _ = process_hemis(train_netmats, train_subIDs, id_df)

    test_meshes_L, test_meshes_R = process_hemis(test_meshes, test_subIDs, id_df)
    test_netmats, _ = process_hemis(test_netmats, test_subIDs, id_df)

    validation_meshes_L, validation_meshes_R = process_hemis(validation_meshes, valid_subIDs, id_df)
    validation_netmats, _ = process_hemis(validation_netmats, valid_subIDs, id_df)

    # concat
    nontrain_meshes_L = concat_dfs(test_meshes_L, validation_meshes_L)
    nontrain_meshes_R = concat_dfs(test_meshes_R, validation_meshes_R)
    nontrain_netmats = concat_dfs(test_netmats, validation_netmats)

    print("Processed all data.")

    # regressions
    train_meshes_L_regression = univariate_regressions(train_meshes_L, pheno_df)
    train_meshes_R_regression = univariate_regressions(train_meshes_R, pheno_df)

    print("Train mesh regressions complete.")

    nontrain_meshes_L_regression = univariate_regressions(nontrain_meshes_L, pheno_df)
    nontrain_meshes_R_regression = univariate_regressions(nontrain_meshes_R, pheno_df)

    print("Nontrain mesh regressions complete.")

    train_netmat_regression = univariate_regressions(train_netmats, pheno_df)
    nontrain_netmat_regression = univariate_regressions(nontrain_netmats, pheno_df)

    print("Netmat regressions complete.")
    
    # save ouputs
    save_path = f"/scratch/ahmadf/NeuroTranslate/SurfToNetmat/regressions/{translation}"

    train_meshes_L_regression.to_csv(f"{save_path}/train_meshes_L_regression.csv")
    train_meshes_R_regression.to_csv(f"{save_path}/train_meshes_R_regression.csv")

    nontrain_meshes_L_regression.to_csv(f"{save_path}/nontrain_meshes_L_regression.csv")
    nontrain_meshes_R_regression.to_csv(f"{save_path}/nontrain_meshes_R_regression.csv")

    train_netmat_regression.to_csv(f"{save_path}/train_netmat_regression.csv")
    nontrain_netmat_regression.to_csv(f"{save_path}/nontrain_netmat_regression.csv")

    print("Saved all ouputs.")

