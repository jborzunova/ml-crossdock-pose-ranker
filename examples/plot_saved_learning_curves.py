import os
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # Добавляем путь к корню проекта (2 уровня вверх от examples/)
from app.plots import *
import joblib

def plot_best_lc(filename):

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n File: {filename}")
    print("Comments:", data.get("comment", ""))
    print("Parameters:")
    for k, v in data.get("parameters", {}).items():
        print(f"   {k}: {v}")
    plot_train_val_lc(data)


def plot_optuna_trials_lc(filename_prefix):
    import json
    import joblib
    from app.plots import plot_all_valid_learning_curves, plot_topk_train_val_learning_curves

    path_prefix = f"data/learning_curves/{filename_prefix}_optuna_trials"

    # === Загрузка learning_curves_by_trial ===
    with open(f"{path_prefix}_lcs.json", "r", encoding="utf-8") as f:
        lcs_data = json.load(f)

    learning_curves_by_trial_raw = lcs_data.get("learning_curves_by_trial", {})

    # Приводим ключи к int
    learning_curves_by_trial = {
        int(k): v for k, v in learning_curves_by_trial_raw.items()
    }

    print(f"\n Файл: {path_prefix}_lcs.json")
    print(" Комментарий:", lcs_data.get("comment", ""))
    print(" Параметры:")
    for k, v in lcs_data.get("parameters", {}).items():
        print(f"   {k}: {v}")

    # === Загрузка Optuna study ===
    study_data = joblib.load(f"{path_prefix}_study.pkl")
    study = study_data["study"]

    # === Рисуем графики ===
    plot_all_valid_learning_curves(learning_curves_by_trial, study)
    plot_topk_train_val_learning_curves(learning_curves_by_trial, study)


if __name__ == "__main__":
    plot_optuna_trials_lc("2025-08-26_12-27") # example for results of Optuna search hyperparameters
    plot_best_lc('data/learning_curves/2025-08-26_12-33_best_params_lc.json')   # example for plot of the best model run
