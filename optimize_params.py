# conda activate ml
from app.optuna_objective import make_objective
from app.preprocessing import *
from app.plots import *
import optuna
import pandas as pd
import json
import joblib

if __name__ == "__main__":
    # ---- Data Preparation ----
    data = pd.read_csv(DATA_PATH, index_col=0)
    data['label'] = data['rmsd'].apply(rmsd_to_relevance)
    data = drop_zero_label_groups(data)  # в обучении и валидации эти данные не нужны. Только в тесте, после
    #data = rmsd2rank(data)  # add columns label and rank (xgbranker works only with discrete y)

    #X_raw = extract_X(data)
    #_, SVD_model = reduce_dim(X_raw)
    #joblib.dump(SVD_model, SVD_MODEL_PATH)  # save for run_best_model
    SVD_model = joblib.load(SVD_MODEL_PATH)  # for speed

    # ---- Data Training and LOLO Evaluation ----
    print("\n=== Optuna Parameters Optimization ===")

    learning_curves_by_trial = {}  # dictionary to store averaged learning curves
    # for different model parameters being optimized by Optuna

    # create the objective function
    # objective(trial) must have access to certain variables
    # but still follow the required Optuna format: objective(trial)
    objective = make_objective(
                    data=data,
                    SVD_model=SVD_model,
                    learning_curves_by_trial=learning_curves_by_trial
                    )

    # ---- Optimize Model Parameters ----
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    with open(PARAMS_PATH, "w") as f:
        json.dump(study.best_params, f)
    print("Best params:", study.best_params)
    print(f"Best {METRIC}:", study.best_value)
    # ---- Plot Learning Curves for Model Parameters Tuning ----
    print('learning_curves_by_trial =', learning_curves_by_trial)
    plot_all_valid_learning_curves(learning_curves_by_trial, study)
    plot_topk_train_val_learning_curves(learning_curves_by_trial, study)
