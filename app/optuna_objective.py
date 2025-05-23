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
        for test_ligand in tqdm(unique_ligands,
                                desc=f"LOLO Evaluation {trial.number}"):
            # ---- Prepare Data ----
            df_train = data[data['init_ligand_file'] != test_ligand].copy()
            df_test = data[data['init_ligand_file'] == test_ligand].copy()
            X_train, y_train, group_train = prepare_XGB_data(df_train, SVD_model)
            X_test, y_test, group_test = prepare_XGB_data(df_test, SVD_model)
            # ---- Validation only on data with at least one native like pose ----
            if 1 in y_test:  # Skip data if does not have any native-like pose. Nothing to rank
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
                            eval_set=[(X_test, y_test)],
                            eval_group=[group_test],
                            verbose=False
                            )
                evals_results.append(list(model.evals_result().values())[0])
                '''
                # ---- Data Retrieval for Threshold Choosing Plot ----
                df_test['score'] = model.predict(X_test)
                df_test['true_label'] = y_test
                top_ranked = df_test.sort_values(by='score', ascending=False).groupby('init_ligand_file').head(1)
                top_score = top_ranked['score'].iloc[0]
                is_native_like = top_ranked['true_label'].iloc[0]
                n_cl = top_ranked['n_cluster'].iloc[0]
                per_ligand_scores.append({
                    'ligand': test_ligand,
                    'top_score': top_score,
                    'top_rank_is_native': is_native_like,
            	    'n_cluster': n_cl
                })
                '''
                # plot_learning_curve(evals_results, fold) # если нужно отследить обучение внутри одного прогона
        # ---- Get the Result of Cross-Validation for Optuna ----
        mean_curve = get_combined_learning_curve(evals_results)
        learning_curves_by_trial[trial.number] = mean_curve
        return mean_curve[-1]
    return objective
