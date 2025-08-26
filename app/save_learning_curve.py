import json
from datetime import datetime
from pathlib import Path
import joblib
import types
import importlib

def save_learning_curve(learning_curves: dict, best_parameters: dict, experiment_name: str = "lc"):
    answer = input("Save learning curves? (y/n): ").strip().lower()
    if answer not in ['y', 'yes']:
        print("Not saving")
        return
    # Комментарий пользователя
    comment = input("Enter a comment to the experiment: ").strip()
    # Финальный словарь
    save_dict = {
        "comment": comment,
        "parameters": best_parameters,
        "train": list(learning_curves.get("train", [])),
        "valid": list(learning_curves.get("valid", [])),
    }
    # Путь для сохранения
    Path("data/learning_curves").mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f"data/learning_curves/{date_str}_{experiment_name}.json"
    # Сохраняем в JSON
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=2, ensure_ascii=False)
    print(f"Data saved in {filename}")


def save_learning_curves_and_study(learning_curves_by_trial: dict, study: object, experiment_name="optuna_trials"):
    answer = input("Save learning_curves_by_trial and study? (y/n): ").strip().lower()
    if answer not in ['y', 'yes']:
        print("Saving is calcelled.")
        return

    comment = input("Enter a comment to the experiment: ").strip()

    now_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    Path("data/learning_curves").mkdir(exist_ok=True)

    try:
        parameters = importlib.import_module("parameters")
        param_dict = {
            key: getattr(parameters, key)
            for key in dir(parameters)
            if not key.startswith("__") and not isinstance(getattr(parameters, key), types.ModuleType)
        }
    except ModuleNotFoundError:
        print("⚠️ Файл parameters.py не найден. Параметры не будут сохранены.")
        param_dict = {}

    data_to_save = {
        "comment": comment,
        "parameters": param_dict,
        "learning_curves_by_trial": learning_curves_by_trial,
    }

    filename_lc = f"data/learning_curves/{now_str}_{experiment_name}_lcs.json"
    with open(filename_lc, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False)
    print(f"learning_curves_by_trial сохранён в {filename_lc}")

    filename = f"data/learning_curves/{now_str}_{experiment_name}_study.pkl"
    joblib.dump({"study": study, "comment": comment}, filename)
    print(f"Optuna study сохранён в {filename}")
