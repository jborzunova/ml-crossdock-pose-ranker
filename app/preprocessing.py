import pandas as pd
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from parameters import *


def change_filename(row):
    return row['file'].replace('.pdb', '')


def read_merge_data(dataset):
    data = pd.read_csv(dataset)
    print(data.columns)
    print('init shape of data:', data.shape)
    data2 = pd.read_csv('/home/oem/research/colchicine_site/data.csv')  # cluster info
    print('shape of docking data:', data2.shape)
    data2 = data2[['file', 'n_cluster', 'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']]
    data2['file'] = data2.apply(change_filename, axis=1)
    data = data.merge(data2, how='left', right_on='file', left_on='init_ligand_file')
    data.drop(columns=['file'], inplace=True)
    print(data.columns)

    #merged_dataset = dataset.replace('.csv', '_trainval.csv')
    #data.to_csv(merged_dataset)
    print('len of data after merge', len(data))
    print('Finished')
    return data


def reduce_dim(X, target_variance = 0.95):
    # ---- Уменьшение размерности данных ----
    # --- Автовыбор n_components по explained variance ---
    print('Reducing Dimensions of Data ...')
    max_components = min(X.shape[0] - 1, X.shape[1])
    # Пробуем много компонент и проверяем накопленную дисперсию
    svd_full = TruncatedSVD(n_components=max_components, random_state=42)
    svd_full.fit(X)
    explained_variance = np.cumsum(svd_full.explained_variance_ratio_)
    n_components_optimal = np.argmax(explained_variance >= target_variance) + 1
    print(f"Выбрано {n_components_optimal} компонент для сохранения {target_variance:.0%} дисперсии.")
    # Теперь применим SVD с оптимальным количеством компонент
    svd = TruncatedSVD(n_components=n_components_optimal, random_state=SEED)
    X_reduced = svd.fit_transform(X)
    print('the new shape of features is', X_reduced.shape)
    return X_reduced, svd


def prepare_XGB_data(df, svd_model, target_variance=0.95):
    # Cортировка необходима для модели ranker!
    # (лист group говорят модели размер последовательных групп)
    df = df.sort_values(by='init_ligand_file').reset_index(drop=True)
    #print(df.loc[:150, 'init_ligand_file'])
    y = df['labels'].values
    X_raw = extract_X(df)
    X_reduced = svd_model.transform(X_raw)
    #X_df = pd.DataFrame(X_reduced, index=df.index)
    #print(type(X_reduced))
    # Считаем размер каждой группы
    group = df.groupby('init_ligand_file').size().tolist()
    return X_reduced, y, group


def extract_X(df):
    return df.drop(columns=['Unnamed: 0', 'labels', 'init_ligand_file', 'n_cluster',
					'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'])


if __name__ == '__main__':
    data = read_merge_data('data_train_crossdock_rmsd_ccf.csv')
    print('I merged the data with n_cluster information, its shape is', data.shape)
