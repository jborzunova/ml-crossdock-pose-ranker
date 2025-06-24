# conda activate ml
import json
import pandas as pd
from xgboost import XGBRanker
from app.preprocessing import *
from app.plots import *
from app.analysis import *
from tqdm import tqdm
from parameters import *
import joblib


if __name__ == "__main__":
    # === Load data ===
    data, _ = data_read_prep()
    SVD_model = joblib.load(SVD_MODEL_PATH)
    ohe_encoder = joblib.load(OHE_ENCODER_PATH)
    # === Load best params ===
    with open(PARAMS_PATH, "r") as f:
        best_params = json.load(f)
    # === Prepare model ===
    print('best_params:', best_params)
    model = XGBRanker(objective=OBJECTIVE, eval_metric=METRIC,
                      random_state=SEED, **best_params,
                      n_estimators=N_ESTIMATORS,
                      early_stopping_rounds=EARLY_STOPPING_ROUNDS
                      )
    # === Evaluate ===
    unique_clusters = data['lig_cluster'].unique()
    evals_results = []
    # ---- LOLO algorithm ----
    for val_cluster in tqdm(unique_clusters, desc=f"LOCO Evaluation"):
        # ---- Prepare Data ----
        df_train = data[data['lig_cluster'] != val_cluster].copy()
        df_val = data[data['lig_cluster'] == val_cluster].copy()
        #print(f'ligands of cluster {val_cluster}:', df_val['ligand'].unique())

        X_train, y_train, group_train = prepare_XGB_data(df_train, SVD_model, ohe_encoder)
        #print('X_train.shape', X_train.shape)
        X_val, y_val, group_val = prepare_XGB_data(df_val, SVD_model, ohe_encoder)
        #print('X_val.shape', X_val.shape)
        model.fit(
                        X_train, y_train,
                        group=group_train,
                        eval_set=[(X_train, y_train), (X_val, y_val)], # Also evaluate on the training set to check for model overfitting
                        eval_group=[group_train, group_val],
                        verbose=False
                  )
        raw_result = model.evals_result()
            # Renaming automatic keys from Optuna manually
        mapped_result = {
                            'train': raw_result.get('validation_0', {}),
                            'valid': raw_result.get('validation_1', {})
                         }
        evals_results.append(mapped_result)

    # ---- Get the Result of Cross-Validation for Optuna ----
    # mean_curves - is 2 Learning Curves (train and validation)
    mean_curves = get_combined_learning_curves(evals_results, metric=METRIC)
    print('learning curves =', mean_curves)
    plot_train_val_lc(mean_curves)
