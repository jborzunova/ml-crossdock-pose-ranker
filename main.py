# conda activate ml
from app.optuna_objective import make_objective
from app.preprocessing import *
from app.plots import *
import optuna

if __name__ == "__main__":
    # ---- Data Preparation ----
    data = read_merge_data('./data_crossdock_rmsd_ccf.csv')
    X_raw = extract_X(data)
    _, SVD_model = reduce_dim(X_raw)

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
    print("Best params:", study.best_params)
    print("Best map@1:", study.best_value)
    # ---- Plot Learning Curves for Model Parameters Tuning ----
    print('learning_curves_by_trial =', learning_curves_by_trial)
    plot_all_valid_learning_curves(learning_curves_by_trial, study)
    plot_topk_train_val_learning_curves(learning_curves_by_trial, study)
