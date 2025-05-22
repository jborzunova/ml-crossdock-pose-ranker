import pandas as pd


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


if __name__ == '__main__':
    data = read_merge_data('data_train_crossdock_rmsd_ccf.csv')
    print('I merged the data with n_cluster information, its shape is', data.shape)
