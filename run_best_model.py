# conda activate pose_ranker
import json
import pandas as pd
from xgboost import XGBRanker
from app.preprocessing import *
from app.plots import *
from app.analysis import *
from app.save_learning_curve import *
from tqdm import tqdm
from parameters import *
import joblib
from sklearn.metrics import ndcg_score
import numpy as np


def random_model_baseline(data):
    """
    Случайная модель для оценки среднего NDCG по LOCO
    """
    unique_clusters = data['lig_cluster'].unique()
    scores = []
    for val_cluster in tqdm(unique_clusters, desc="Random baseline LOCO"):
        # ---- Prepare Data ----
        X_train, y_train, group_train, weights_train, \
        X_val, y_val, group_val = get_sets(data, val_cluster)
        # --- Случайные предсказания ---
        y_pred = np.random.rand(len(y_val))
        # --- ndcg_score требует 2D массивы ---
        ndcg = ndcg_score([y_val], [y_pred])
        scores.append(ndcg)
    mean_ndcg = np.mean(scores)
    print(f"Случайная модель (mean NDCG): {mean_ndcg:.4f}")
    return mean_ndcg


if __name__ == "__main__":
    # === Load data ===
    data = data_read_prep()
    random_model_baseline(data)
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
    # ---- LOLO algorithm ----
    unique_clusters = data['lig_cluster'].unique()
    evals_results = []
    all_importances = []  # feature importance
    for val_cluster in tqdm(unique_clusters, desc=f"LOCO Evaluation"):
        # ---- Prepare Data ----
        X_train, y_train, group_train, weights_train, \
        X_val, y_val, group_val = get_sets(data, val_cluster)
        #print('train set:', X_train.shape, y_train.shape, group_train)
        #print('val set', X_val.shape, y_val.shape)
        #print('group_val:')
        #print(group_val)
        model.fit(
                        X_train, y_train,
                        group=group_train,
                        sample_weight=weights_train,
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


        # ---- Извлечь важности ----
        importance = model.get_booster().get_score(importance_type='gain')
        # Преобразуем в Series и добавим в список
        imp_series = pd.Series(importance, name=val_cluster)
        all_importances.append(imp_series)

    # ---- Get the Result of Cross-Validation for Optuna ----
    # mean_curves - is 2 Learning Curves (train and validation)
    plot_feature_importance(all_importances)
    mean_curves = get_combined_learning_curves(evals_results, metric=METRIC)
    print('learning curves =', mean_curves)
    save_learning_curve(mean_curves, best_params, 'best_params_lc')
    plot_train_val_lc(mean_curves)
