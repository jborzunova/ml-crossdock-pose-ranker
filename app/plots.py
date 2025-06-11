import matplotlib.pyplot as plt
import matplotlib.cm as cm
from parameters import *
import numpy as np


def plot_all_valid_learning_curves(learning_curves_by_trial, study):
    """
    Plot all validation learning curves obtained from Optuna trials.

    Parameters:
    - learning_curves_by_trial: dict[trial_num] = {'train': evals, 'valid': evals}
    - study: Optuna study object used to access the best_trial
    """
    plt.figure(figsize=(10, 5))

    for trial_num, curve_dict in learning_curves_by_trial.items():
        valid_curve = curve_dict.get("valid", {})
        if valid_curve is None:
            continue
        color = 'red' if trial_num == study.best_trial.number else 'gray'
        plt.plot(valid_curve, color=color, linewidth=2, alpha=0.8, label=f"Trial {trial_num}" if trial_num == study.best_trial.number else None)

    plt.title(f"Validation Learning Curves (metric: {METRIC})")
    plt.xlabel("Boosting Iterations")
    plt.ylabel(METRIC)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # ---- The best Learning Curve ---
    best_trial_num = study.best_trial.number
    best_curve = learning_curves_by_trial[best_trial_num]['valid']
    print(f"\nBest Trial Number: {best_trial_num}")
    print(f"Best Trial Validation Curve: {np.round(best_curve, 3)}")
    plt.savefig("images/all_valid_learning_curves.png", dpi=300)
    plt.show()



def plot_topk_train_val_learning_curves(learning_curves_by_trial, study, top_k=5):
    """
    Plots Learning Curves for train/valid datasets for top-k trials.
    The best model (the best performance on validation sets) is drawn in red.
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

    plt.xlabel("Boosting iterations")
    plt.ylabel("map@1")
    plt.title(f"Top-{top_k} Learning Curves (train & valid) - XGBRanker")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"images/topk_learning_curves.png", dpi=300)
    plt.show()
