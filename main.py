# conda activate ml
from app.optuna_objective import make_objective
from app.preprocessing import *
from app.plots import *
import optuna

if __name__ == "__main__":
    # ---- Data Preparation ----
    data = read_merge_data('./data_train_crossdock_rmsd_ccf.csv')  # Загрузка данных
    X_raw = extract_X(data)
    _, SVD_model = reduce_dim(X_raw)

    # ---- Data Training and LOLO Evaluation ----
    print("\n=== Optuna Parameters Optimization ===")

    learning_curves_by_trial = {}  # словарь для сохранения средних кривых обучения
    # для разных параметров модели (которые оптимизируются в процессе работы Optuna)

    # создаём objective-функцию
    # objective(trial) должна иметь доступ к некоторым переменным, но при этом
    # должна оставаться в формате objective(trial), как требует Optuna
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
    plot_combined_learning_curves(learning_curves_by_trial, study)


'''
# ---- Plot the Results for Threshold Choose ----
# Sort by score (optional, just for a nice x-axis order)
sorted_scores = sorted(per_ligand_scores, key=lambda x: x['top_score'], reverse=True)

scores = [d['top_score'] for d in sorted_scores]
colors = ['green' if d['top_rank_is_native'] else 'red' for d in sorted_scores]
labels = [d['n_cluster'] for d in sorted_scores]

'''
