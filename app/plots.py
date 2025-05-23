import matplotlib.pyplot as plt
import matplotlib.cm as cm
from parameters import *
import numpy as np


def plot_learning_curve(evals_results, fold):
    '''
    Эта функция рисует отдельные Learning Curve для валидации на 1 лиганде
    '''
    for metric in evals_results[f'validation_{fold}'].keys():
        val_map = evals_results[f'validation_{fold}'][metric]
        print('val_map =', val_map)
        epochs = range(1, len(val_map) + 1)
        plt.figure(figsize=(8, 4))
        plt.title("XGBRanker Learning Curve")
        plt.plot(epochs, val_map,
            label=f"Fold {fold+1}")
        plt.xlabel("Boosting Iterations")
        plt.ylabel(metric)
        plt.grid(True)
        plt.show()



def plot_all_valid_learning_curves(learning_curves_by_trial, study):
    """
    Строим график всех валидационных кривых обучения, полученных в Optuna.

    Parameters:
    - learning_curves_by_trial: dict[trial_num] = {'train': evals, 'valid': evals}
    - study: объект Optuna для доступа к best_trial
    """
    plt.figure(figsize=(10, 5))

    for trial_num, curve_dict in learning_curves_by_trial.items():
        valid_curve = curve_dict.get("valid", {})
        if valid_curve is None:
            continue
        color = 'red' if trial_num == study.best_trial.number else 'gray'
        plt.plot(valid_curve, color=color, linewidth=2, alpha=0.8, label=f"Trial {trial_num}" if trial_num == study.best_trial.number else None)

    plt.title(f"Кривые обучения на валидации (метрика: {METRIC})")
    plt.xlabel("Boosting итерации")
    plt.ylabel(METRIC)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---- Вывод лучшей кривой обучения ---
    best_trial_num = study.best_trial.number
    best_curve = learning_curves_by_trial[best_trial_num]['valid']
    print(f"\nBest Trial Number: {best_trial_num}")
    print(f"Best Trial Validation Curve: {np.round(best_curve, 3)}")

    plt.show()



def plot_topk_train_val_learning_curves(learning_curves_by_trial, study, top_k=5):
    """
    Рисует кривые обучения train/valid для топ-k триалов.
    Лучшая модель (по валидации) выделяется красным цветом и жирной линией.
    """
    best_trials = sorted(
        study.trials, key=lambda t: t.value if t.value is not None else float("inf")
    )[:top_k]
    best_trial_number = study.best_trial.number

    plt.figure(figsize=(12, 6))

    colormap = cm.get_cmap('tab10', top_k)

    for idx, trial in enumerate(best_trials):
        trial_num = trial.number
        curves = learning_curves_by_trial.get(trial_num)
        if not curves:
            continue

        is_best = trial_num == best_trial_number
        color = 'red' if is_best else colormap(idx)
        lw = 2.0 if is_best else 1.5
        zorder = 10 if is_best else 1  # Лучший поверх остальных

        for set_name in ['train', 'valid']:
            curve = curves.get(set_name)
            if curve is None:
                continue
            linestyle = '--' if set_name == 'train' else '-'
            label = (
                f"Best Trial {trial_num} - {set_name}"
                if is_best else f"Trial {trial_num} - {set_name}"
            )
            plt.plot(curve, label=label, linestyle=linestyle, color=color, linewidth=lw, alpha=0.9, zorder=zorder)

    plt.xlabel("Boosting итерации")
    plt.ylabel("map@1")
    plt.title(f"Top-{top_k} Learning Curves (train & valid) - XGBRanker")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
'''
# ---- Plot the Results for Threshold Choose ----
plt.figure(figsize=(14, 5))
bars = plt.bar(range(len(scores)), scores, color=colors)
plt.axhline(y=0.5, color='gray', linestyle='--', label='Score threshold (0.5)')
plt.title("Top-ranked Pose Score per Ligand")
plt.xlabel("Ligand Index (sorted by score)")
plt.ylabel("Top-ranked Pose Score")
plt.legend()
# Optional: Add a few x-tick labels
xtick_indices = np.linspace(0, len(labels)-1, num=10, dtype=int)
plt.xticks(xtick_indices, [labels[i] for i in xtick_indices], rotation=45)
plt.tight_layout()
plt.show()
'''
