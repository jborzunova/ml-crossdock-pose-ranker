from parameters import *
import numpy as np


def get_combined_learning_curve(all_evals_results, metric=METRIC):
    """
    all_evals_results: список словарей с результатами обучения (по фолдам)
                       каждый элемент соответствует одному фолду
                       и имеет структуру как evals_result из XGBRanker.fit()
    metric: название метрики, которую нужно визуализировать (например, 'map@1')
    """
    # Собираем все кривые
    all_curves = []
    for fold_idx, evals in enumerate(all_evals_results):
        if metric not in evals.keys():
            print(f"Фолд {fold_idx}: метрика {metric} не найдена")
            continue
        curve = evals[metric]
        all_curves.append(curve)
    # Приводим к numpy и выравниваем по длине (если нужно)
    max_len = min(len(curve) for curve in all_curves)  # лучше брать min, чтобы не обрезать данные
    trimmed_curves = np.array([c[:max_len] for c in all_curves])
    mean_curve = np.mean(trimmed_curves, axis=0)
    return mean_curve
