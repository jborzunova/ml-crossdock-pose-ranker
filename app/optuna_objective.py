from app.preprocessing import *
from app.analysis import *
from xgboost import XGBRanker
from tqdm import tqdm
from parameters import *


def make_objective(data, SVD_model, learning_curves_by_trial):
    def objective(trial):
        '''
        This function helps find the best parameters for ML model
        It trains the model and evaluate it on validation set with LOLO algorithm.
        Then all Learning Curves (validation on each ligand) are gathered to obtain
        a mean Learning Curve. The output of this function is the last point of it.
        Based on this value the Optuna will decide which parameters give the best
        result
        '''
        unique_ligands = data['init_ligand_file'].unique()
        evals_results = []
        # ---- LOLO algorithm ----
        for val_ligand in tqdm(unique_ligands[:3],
                                desc=f"LOLO Evaluation {trial.number}"):
            # ---- Prepare Data ----
            df_train = data[data['init_ligand_file'] != val_ligand].copy()
            df_val = data[data['init_ligand_file'] == val_ligand].copy()
            X_train, y_train, group_train = prepare_XGB_data(df_train, SVD_model)
            X_val, y_val, group_val = prepare_XGB_data(df_val, SVD_model)
            # ---- Validation only on data with at least one native like pose ----
            if 1 in y_val:  # Skip data if does not have any native-like pose. Nothing to rank
                params = {
                            'objective': 'rank:ndcg',
                            'eval_metric': METRIC,
                            'learning_rate': trial.suggest_float('learning_rate', *LEARNING_RATE_RANGE, log=True),
                            'max_depth': trial.suggest_int('max_depth', *MAX_DEPTH_RANGE),
                            'min_child_weight': trial.suggest_int('min_child_weight', *MIN_CHILD_WEIGHT),
                            'gamma': trial.suggest_float('gamma', *GAMMA_RANGE),
                            'subsample': trial.suggest_float('subsample', *SUBSAMPLE_RANGE),
                            'colsample_bytree': trial.suggest_float('colsample_bytree', *COLSAMPLE_BYTREE),
                            'n_estimators': N_ESTIMATORS,
                            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
                            'random_state': SEED
                          }
                model = XGBRanker(**params)
                model.fit(
                            X_train, y_train,
                            group=group_train,
                            eval_set=[(X_train, y_train), (X_val, y_val)], # сделать оценку на трейне тоже, чтобы проверить переобучаемость модели
                            eval_group=[group_train, group_val],
                            verbose=False
                            )
                raw_result = model.evals_result()
                # Переименование ключей вручную
                mapped_result = {
                    'train': raw_result.get('validation_0', {}),
                    'valid': raw_result.get('validation_1', {})
                }
                evals_results.append(mapped_result)
                '''
                # ---- Data Retrieval for Threshold Choosing Plot ----
                df_val['score'] = model.predict(X_val)
                df_val['true_label'] = y_val
                top_ranked = df_val.sort_values(by='score', ascending=False).groupby('init_ligand_file').head(1)
                top_score = top_ranked['score'].iloc[0]
                is_native_like = top_ranked['true_label'].iloc[0]
                n_cl = top_ranked['n_cluster'].iloc[0]
                per_ligand_scores.append({
                    'ligand': val_ligand,
                    'top_score': top_score,
                    'top_rank_is_native': is_native_like,
            	    'n_cluster': n_cl
                })
                '''
                # plot_learning_curve(evals_results, fold) # если нужно отследить обучение внутри одного прогона
        # ---- Get the Result of Cross-Validation for Optuna ----
        # mean_curves - это 2 кривые обучения (оцененные по трейну или по валидационному сетам и усредненные по кросс-валидации)
        mean_curves = get_combined_learning_curves(evals_results, metric="map@1")
        learning_curves_by_trial[trial.number] = mean_curves
        return mean_curves['valid'][-1]  # оптимизатор опирается на метрику, оуененную по валидационному сету
    return objective
