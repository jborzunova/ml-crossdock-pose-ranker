import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from parameters import *


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


'''
def rmsd2rank(df):
    """
    Adds two columns to the dataframe:
    - 'rank': rank of each pose within its group (lower RMSD = higher rank)
    - 'label': 1 if RMSD <= 2.0 Å, otherwise 0

    Ranking is done within each ligand group.
    """
    df = df.copy()

    # Assign label: 1 if RMSD <= 2 Å, else 0
    df['label'] = (df['rmsd'] <= 2.0).astype(int)

    # Rank poses within each ligand group by RMSD (lower RMSD = better rank)
    # Ranking starts from 1; equal values receive the same rank (dense ranking)
    df['rank'] = df.groupby('ligand')['rmsd'].rank(method='dense', ascending=True).astype(int)

    return df
'''

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


def prepare_XGB_data(df, svd_model, target_variance=0.95):
    # Sorting is required for the ranker model!
    # (The 'group' list tells the model the size of sequential groups)
    df = df.sort_values(by='ligand').reset_index(drop=True)

    y = df['label'].values
    X_raw = extract_X(df)
    X_reduced = svd_model.transform(X_raw)

    # Calculate the size of each group
    group = df.groupby('ligand').size().tolist()
    return X_reduced, y, group


def extract_X(df):
    return df.drop(columns=['ligand', 'protein', 'rmsd', 'label',
                            'mol_weight', 'lig_cluster', 'pr_cluster',
                            'docking_type'])
