from app.preprocessing import *
from app.analysis import *
from xgboost import XGBRanker
from tqdm import tqdm
from parameters import *


def make_objective(data, SVD_model, learning_curves_by_trial):
    '''
    This function helps find the best parameters for ML model
    It trains the model and evaluate it on validation set with LOLO algorithm.
    Then all Learning Curves (validation on each ligand) are gathered to obtain
    a mean Learning Curve. The output of this function is the last point of it.
    Based on this value the Optuna will decide which parameters give the best
    result
    '''
    def objective(trial):
        unique_ligands = data['ligand'].unique()
        evals_results = []
        # ---- LOLO algorithm ----
        for val_ligand in tqdm(unique_ligands,
                                desc=f"LOLO Evaluation {trial.number}"):
            #print(val_ligand)
            # ---- Prepare Data ----
            df_train = data[data['ligand'] != val_ligand].copy()
            df_val = data[data['ligand'] == val_ligand].copy()
            X_train, y_train, group_train = prepare_XGB_data(df_train, SVD_model)
            X_val, y_val, group_val = prepare_XGB_data(df_val, SVD_model)
            #print(X_train.shape, y_train.shape, len(group_train))
            #print(X_val.shape, y_val.shape, len(group_val))
            # ---- Validation only on data with at least one native like pose ----
            #print('y_val', y_val)
            params = {
                        'objective': OBJECTIVE,
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
                            eval_set=[(X_train, y_train), (X_val, y_val)], # Also evaluate on the training set to check for model overfitting
                            eval_group=[group_train, group_val],
                            verbose=False
                        )
            raw_result = model.evals_result()
            #print('.............................')
            #print('raw_result', raw_result)
            #print('.............................')
            # Renaming automatic keys from Optuna manually
            mapped_result = {
                    'train': raw_result.get('validation_0', {}),
                    'valid': raw_result.get('validation_1', {})
                }
            evals_results.append(mapped_result)

        # ---- Get the Result of Cross-Validation for Optuna ----
        # mean_curves - is 2 Learning Curves
        print('evals_results', evals_results)
        mean_curves = get_combined_learning_curves(evals_results, metric=METRIC)
        print('mean_curves =', mean_curves)
        learning_curves_by_trial[trial.number] = mean_curves
        return mean_curves['valid'][-1]  # The optimizer relies on the metric evaluated on the validation set
    return objective
