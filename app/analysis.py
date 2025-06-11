from parameters import *
import numpy as np


def get_combined_learning_curves(all_evals_results, metric="map@1"):
    """
    Returns averaged learning curves across folds for both 'train' and 'valid'.

    Parameters:
    - all_evals_results: list of dictionaries, each with the structure:
                         {'train': {'metric': [...]}, 'valid': {'metric': [...]}}

    - metric: name of the metric, e.g., 'map@1'

    Returns:
    - A dictionary with keys 'train' and 'valid', each containing the averaged curve.
    """
    combined_curves = {}
    for set_name in ["train", "valid"]:
        all_curves = []
        for fold_idx, evals in enumerate(all_evals_results):
            if set_name not in evals or metric not in evals[set_name]:
                print(f"Фолд {fold_idx}: метрика {metric} в {set_name} не найдена")
                continue
            curve = evals[set_name][metric]
            all_curves.append(curve)

        if not all_curves:
            combined_curves[set_name] = None
            continue

        min_len = min(len(c) for c in all_curves)
        trimmed_curves = np.array([c[:min_len] for c in all_curves])
        mean_curve = np.mean(trimmed_curves, axis=0)
        combined_curves[set_name] = mean_curve

    return combined_curves
