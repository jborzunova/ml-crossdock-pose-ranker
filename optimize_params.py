# conda activate ml
from app.optuna_objective import make_objective
from app.preprocessing import *
from app.plots import *
from app.save_learning_curve import *
import optuna
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import json
import joblib


if __name__ == "__main__":
    # ---- Data Preparation ----
    data = data_read_prep()
    
    # ---- Data Training and LOLO Evaluation ----
    print("\n=== Optuna Parameters Optimization ===")

    learning_curves_by_trial = {}  # dictionary to store averaged learning curves
    # for different model parameters being optimized by Optuna

    objective = make_objective(
                                data=data,
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
    save_learning_curves_and_study(learning_curves_by_trial, study)
    plot_all_valid_learning_curves(learning_curves_by_trial, study)
    plot_topk_train_val_learning_curves(learning_curves_by_trial, study)
