import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from parameters import *
import os


def data_read_prep():
    data = pd.read_csv(DATA_PATH, index_col=0)
    data['label'] = data['rmsd'].apply(rmsd_to_relevance)
    data = drop_zero_label_groups(data)  # в обучении и валидации эти данные не нужны. Только в тесте, после
    return data


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
    Drops all groups from the dataframe where all labels of cross-docking are zero.

    Parameters:
        data (pd.DataFrame): Input DataFrame with at least 'group_id' and 'label' columns.
        group_col (str): Name of the group column.
        label_col (str): Name of the label column.

    Returns:
        pd.DataFrame: Filtered DataFrame with only groups that have at least one non-zero label.
    """
    cross_data = data[data['docking_type'] == 'cross']
    # Identify groups with at least one non-zero label
    valid_groups = cross_data.groupby(group_col)[label_col].apply(lambda x: (x != 0).any())
    # Filter to keep only valid groups
    valid_group_ids = valid_groups[valid_groups].index
    return data[data[group_col].isin(valid_group_ids)].reset_index(drop=True)


def reduce_dim(data, target_variance=TARGET_VARIANCE):
    # ---- Dimensionality reduction ----
    # --- Auto-select n_components based on explained variance ---
    X = extract_ccf(data)
    #print(X.columns)
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


def prepare_XGB_data(df, svd_model):
    # Sort the data for ranking
    df = df.sort_values(by='ligand').reset_index(drop=True)
    # Target and group
    y = df['label'].values
    group = df.groupby('ligand').size().tolist()
    # === Fingerprint features reduced by SVD ===
    ccf_raw = extract_ccf(df)  # fingerprint columns only
    ccf_reduced = svd_model.transform(ccf_raw)
    # создаём имена для SVD-компонент
    svd_feature_names = [f"svd_{i}" for i in range(ccf_reduced.shape[1])]
    # === Молекулярные признаки ===
    mol_weight_features = df[['mol_weight_ligand', 'mol_weight_native']].values
    mol_weight_names = ['mol_weight_ligand', 'mol_weight_native']

    additional_features_names = ['vina_score','cnn_score','gauss0','gauss3',
                              'repulsion','hydrophobic','hbond','hydrophobic_interactions',
                              'water_bridges','salt_bridges','pi_stacks',
                              'pi_cation_interactions','halogen_bonds',
                              'metal_complexes']
    additional_features = df[additional_features_names].values

    delta_mol_weight = (df['mol_weight_ligand'] - df['mol_weight_native']).values.reshape(-1, 1)
    abs_delta_mol_weight = np.abs(df['mol_weight_ligand'] - df['mol_weight_native']).values.reshape(-1, 1)
    delta_names = ['delta_mol_weight', 'abs_delta_mol_weight']
    # === Объединяем ===
    X_combined = np.hstack([ccf_reduced, mol_weight_features, delta_mol_weight, abs_delta_mol_weight, additional_features])
    # создаём DataFrame с понятными именами
    feature_names = svd_feature_names + mol_weight_names + delta_names + additional_features_names
    X_combined = pd.DataFrame(X_combined, columns=feature_names)
    # === Calculate weights based on lig_cluster frequency ===
    cluster_counts = df['lig_cluster'].value_counts() / 88
    df['weight'] = 1.0 / df['lig_cluster'].map(cluster_counts)
    df['weight'] /= df['weight'].mean()  # normalize

    group_weights = df.groupby('ligand')['weight'].mean()
    return X_combined, y, group, group_weights


def extract_ccf(df):
    # Select only columns that are named with digits, like '0', '1', '2' ...
    return df[[col for col in df.columns if col.isdigit()]]


def get_sets(data, val_cluster):
    '''
    SVD модель должна видеть только тренировочные данные. Поэтому ждя каждого кластера
    нужно прогнать эти вычисления в отдельности. Далее если выборки с сокращенной
    размерностью уже сохранены в папке, то вычислять заново не нужно
    '''
    folder = f"data/processed/cluster_{val_cluster}"
    os.makedirs(folder, exist_ok=True)

    paths = {
        "X_train": f"{folder}/X_train.csv",
        "y_train": f"{folder}/y_train.csv",
        "group_train": f"{folder}/group_train.csv",
        "weights_train": f"{folder}/weights_train.csv",
        "X_val": f"{folder}/X_val.csv",
        "y_val": f"{folder}/y_val.csv",
        "group_val": f"{folder}/group_val.csv",
    }

    if all(os.path.exists(p) for p in paths.values()):
        X_train = pd.read_csv(paths["X_train"], index_col=0)
        y_train = pd.read_csv(paths["y_train"], index_col=0).squeeze()
        group_train = pd.read_csv(paths["group_train"], index_col=0).squeeze().tolist()
        weights_train = pd.read_csv(paths["weights_train"], index_col=0).squeeze().tolist()

        X_val = pd.read_csv(paths["X_val"], index_col=0)
        y_val = pd.read_csv(paths["y_val"], index_col=0).squeeze()
        group_val = pd.read_csv(paths["group_val"], index_col=0).squeeze()
        if isinstance(group_val, pd.Series):
            group_val = group_val.tolist()
        else:
            group_val = [group_val]  # если это 1 число, то будет список с 1 элементом. Например, 1 лиганд в тесте и 90 поз. Будет [90]
    else:
        df_train = data[data['lig_cluster'] != val_cluster].copy()
        df_val = data[data['lig_cluster'] == val_cluster].copy()
        _, SVD_model = reduce_dim(df_train)
        X_train, y_train, group_train, weights_train = prepare_XGB_data(df_train, SVD_model)
        #df_val = df_val[df_val['docking_type'] == 'cross']  # for test in run_best_model we take only cross data
        X_val, y_val, group_val, _ = prepare_XGB_data(df_val, SVD_model)

        pd.DataFrame(X_train).to_csv(paths["X_train"])
        pd.Series(y_train).to_csv(paths["y_train"])
        pd.Series(group_train).to_csv(paths["group_train"])
        pd.Series(weights_train).to_csv(paths["weights_train"])

        pd.DataFrame(X_val).to_csv(paths["X_val"])
        pd.Series(y_val).to_csv(paths["y_val"])
        pd.Series(group_val).to_csv(paths["group_val"])
    return X_train, y_train, group_train, weights_train, X_val, y_val, group_val
