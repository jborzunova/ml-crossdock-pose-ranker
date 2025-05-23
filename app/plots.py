import matplotlib.pyplot as plt
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


def plot_combined_learning_curves(learning_curves_by_trial, study, metric=METRIC):
    # Строим график для всех кривых обучения, полученных в цикле Optuna
    plt.figure(figsize=(10, 5))
    for trial_num, curve in learning_curves_by_trial.items():
        color = 'red' if trial_num == study.best_trial.number else 'gray'
        plt.plot(curve, color=color, linewidth=2, alpha=0.8)
    #plt.plot(range(1, max_len+1), combined_curve, color='red', linewidth=2, label='Средняя кривая для кросс-валидации')
    plt.title(f'Кривая обучения XGBRanker (метрика: {metric})')
    plt.xlabel('Boosting итерации')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # ---- Вывод лучшей кривой обучения ---
    best_trial_num = study.best_trial.number
    best_curve = learning_curves_by_trial[best_trial_num]
    print(f"\nBest Trial Number: {best_trial_num}")
    print(f"Best Trial Learning Curve: {np.round(best_curve, 3)}")
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
