import pandas as pd
import numpy as np
from xgboost import XGBRanker
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, classification_report
from read_merge_data import *
import plotext as plt

SEED = 2007
N_SPLITS = 5

# Load and inspect data
data = read_merge_data('./data_train_crossdock_rmsd_ccf.csv')
print("Columns:", data.columns)
print("Shape:", data.shape)
print("Unique ligands:", data['init_ligand_file'].nunique())
print("Label distribution:\n", data['labels'].value_counts())

# Convert features to matrix
data_features = data.drop(columns=['labels', 'init_ligand_file', 'n_cluster', 'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']) 
features = np.vstack(data_features.values)  # assumes 'features' column contains arrays
labels = data['labels'].values
groups = data['n_cluster'].values  # cluster-based CV

group_kfold = GroupKFold(n_splits=N_SPLITS)
evals_results = {}

for fold, (train_idx, test_idx) in enumerate(group_kfold.split(features, labels, groups=groups)):
    print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")

    X_train, y_train = features[train_idx], labels[train_idx]
    X_test, y_test = features[test_idx], labels[test_idx]

    df_train = data.iloc[train_idx].copy()
    df_test = data.iloc[test_idx].copy()

    group_train = df_train.groupby('init_ligand_file').size().tolist()
    group_test = df_test.groupby('init_ligand_file').size().tolist()

    model = XGBRanker(
        objective='rank:map',
        eval_metric='map',
        learning_rate=0.1,
        max_depth=6,
        n_estimators=100,
        random_state=SEED,
        min_child_weight=0.1
        )
    eval_set = [(X_test, y_test)]
    model.fit(
        X_train, y_train,
        group=group_train,
        eval_set=eval_set,
        eval_group=[group_test],
        #eval_metric='map',
        #verbose=True,
        #callbacks=[],
        #evals_result=evals_result
        )
    
    # --- Get the Results for Plot ---
    #print(model.evals_result())
    evals_results[f'validation_{fold}'] = list(model.evals_result().values())[0]
    
    df_test['score'] = model.predict(X_test)
    df_test['true_label'] = y_test
    print('Test:')
    # Group by ligand, select top scoring pose
    top_ranked = df_test.sort_values(by='score', ascending=False).groupby('init_ligand_file').head(1)

    # Now evaluate: how many of these top poses are native-like?
    n_native_like = top_ranked['true_label'].sum()
    total_ligands = top_ranked.shape[0]
    accuracy = n_native_like / total_ligands

    print(f"\nPer-ligand Top-1 Accuracy (Native-like & Top 1): {accuracy:.2f}") 
    print(f"The number of good predicted ligands: ({n_native_like}/{total_ligands})")


# --- Ploting Learing Curve ---
#print(evals_results) 

for fold in range(len(evals_results)):
    # Extract validation MAP curve (validation_0 is test set since you passed eval_set=[(X_test, y_test)])
    val_map = evals_results[f'validation_{fold}']['map']
    epochs = range(1, len(val_map) + 1)
    
    plt.plotsize(50, 20)
    plt.title("XGBRanker Learning Curve")
    plt.plot(epochs, val_map,
        label=f"Fold {fold+1}")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Mean Average Precision (MAP)")
    plt.grid(True)
    plt.show()
    plt.clear_figure()

