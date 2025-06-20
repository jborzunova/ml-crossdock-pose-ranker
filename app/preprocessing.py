import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from parameters import *

def data_read_prep():
    data = pd.read_csv(DATA_PATH, index_col=0)
    data['label'] = data['rmsd'].apply(rmsd_to_relevance)
    data = drop_zero_label_groups(data)  # в обучении и валидации эти данные не нужны. Только в тесте, после
    #print('data.columns', data.columns)
    ccf_data = extract_ccf(data)
    #print('ccf_data.columns', ccf_data.columns)
    return data, ccf_data


def ohe_receptor(data, ohe_encoder):
    ohe_features = ohe_encoder.transform(data[['protein']])
    ohe_feature_names = ohe_encoder.get_feature_names_out(['protein'])
    ohe_df = pd.DataFrame(ohe_features, columns=ohe_feature_names, index=data.index)
    data_combined = pd.concat([data.reset_index(drop=True), ohe_df], axis=1)
    data_combined = data_combined.drop(columns=['protein'])
    #print('data columns after ohe:', data_combined.columns)
    return data_combined


def rmsd_to_relevance(rmsd: float) -> int:
    """Convert RMSD value to an integer relevance label for ranking."""
    if rmsd < 2.0:
        return 2  # highly relevant
    elif rmsd < 3.0:
        return 1  # relevant
    else:
        return 0  # weakly relevant


def drop_zero_label_groups(data: pd.DataFrame, group_col: str = 'ligand', label_col: str = 'label') -> pd.DataFrame:
    """
    Drops all groups from the dataframe where all labels are zero.

    Parameters:
        data (pd.DataFrame): Input DataFrame with at least 'group_id' and 'label' columns.
        group_col (str): Name of the group column.
        label_col (str): Name of the label column.

    Returns:
        pd.DataFrame: Filtered DataFrame with only groups that have at least one non-zero label.
    """
    # Identify groups with at least one non-zero label
    valid_groups = data.groupby(group_col)[label_col].apply(lambda x: (x != 0).any())

    # Filter to keep only valid groups
    valid_group_ids = valid_groups[valid_groups].index
    return data[data[group_col].isin(valid_group_ids)].reset_index(drop=True)


def reduce_dim(X, target_variance=0.95):
    # ---- Dimensionality reduction ----
    # --- Auto-select n_components based on explained variance ---
    print('Reducing Dimensions of Data ...')
    max_components = min(X.shape[0] - 1, X.shape[1])

    # Try many components and check the cumulative explained variance
    svd_full = TruncatedSVD(n_components=max_components, random_state=42)
    svd_full.fit(X)
    explained_variance = np.cumsum(svd_full.explained_variance_ratio_)
    n_components_optimal = np.argmax(explained_variance >= target_variance) + 1
    print(f"{n_components_optimal} components selected to preserve {target_variance:.0%} of variance.")

    # Now apply SVD with the optimal number of components
    svd = TruncatedSVD(n_components=n_components_optimal, random_state=SEED)
    X_reduced = svd.fit_transform(X)
    print('The new shape of features is', X_reduced.shape)
    return X_reduced, svd


def prepare_XGB_data(df, svd_model, ohe_encoder, target_variance=0.95):
    # Sort the data for ranking
    df = df.sort_values(by='ligand').reset_index(drop=True)
    # Apply one-hot encoding to receptor_id
    df = ohe_receptor(df, ohe_encoder)
    # Target and group
    y = df['label'].values
    group = df.groupby('ligand').size().tolist()
    # Extract the fingerprint features and apply SVD
    ccf_raw = extract_ccf(df)  # should return only fingerprint columns
    ccf_reduced = svd_model.transform(ccf_raw)
    # Extract one-hot encoded receptor columns (starts with "receptor_")
    receptor_features = df.filter(regex='^protein_').values
    # Concatenate SVD-reduced fingerprint and receptor one-hot features
    X_combined = np.hstack([ccf_reduced, receptor_features])
    return X_combined, y, group


def extract_ccf(df):
    # Select only columns that are named with digits, like '0', '1', '2' ...
    return df[[col for col in df.columns if col.isdigit()]]
