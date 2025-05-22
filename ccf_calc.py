import os
import numpy as np
import pandas as pd
import deepchem as dc
import time
from concurrent.futures import ThreadPoolExecutor
import logging

SIZE = 1024
PATH_TO_DOCK_FOLDER = '/home/jborzunova/colchicine_site/docking/cross_dock/train0_unidock_default_params/'
DF_NAME = 'data_train_crossdock_rmsd.csv'
DF_NEW_NAME = DF_NAME[:-4] + '_ccf.csv'

data = pd.read_csv(PATH_TO_DOCK_FOLDER + DF_NAME)

EXPECTED_SHAPE = (SIZE*2,)
LOG_FILE = 'errors.log'

# === Logging setup ===
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def ccf_calc(row):
    lig_file = row['path_to_dock_file'] + row['dock_file']
    rec_file = row['path_to_protein'] + row['protein_file'].replace('.pdbqt', '.pdb')
    logging.info(f'Processing {lig_file} and {rec_file}')
    try:
        f = dc.feat.ContactCircularFingerprint(size=SIZE)
        features = f.featurize([(lig_file, rec_file)])  
        features = np.squeeze(features)

        if features.shape != EXPECTED_SHAPE:
            logging.warning(f"[SHAPE ERROR] {lig_file}, {rec_file} → shape: {features.shape}, expected {EXPECTED_SHAPE}")
            return None
        return features
    except Exception as e:
        logging.error(f"[ERROR] Failed on {lig_file}, {rec_file} → {e}")
        return None

#data = data.iloc[:10, :]  # for testing

# Run featurization
start_time = time.time()
with ThreadPoolExecutor(max_workers=20) as executor:
    features = list(executor.map(ccf_calc, [row for _, row in data.iterrows()]))

# Filter out failed entries
valid_features = []
valid_labels = []
valid_init_files = []

for i, f in enumerate(features):
    if isinstance(f, np.ndarray) and f.shape == EXPECTED_SHAPE:
        valid_features.append(f)
        valid_labels.append(1 if data.loc[i, 'rmsd'] <= 2.5 else 0)
        valid_init_files.append(data.loc[i, 'init_ligand_file'])
    else:
        logging.info(f"[SKIPPED] Index {i} has invalid or missing feature.")

# Save result
features_array = np.array(valid_features)
df = pd.DataFrame(features_array)
df['labels'] = valid_labels
df['init_ligand_file'] = valid_init_files
df.to_csv(DF_NEW_NAME, index=False)

print(f"[INFO] Saved {len(df)} valid entries to {DF_NEW_NAME}")
print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
print(f"[INFO] See {LOG_FILE} for details on skipped or failed featurizations.")
