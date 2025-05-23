from parameters import *
import numpy as np

'''
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
'''

def get_combined_learning_curves(all_evals_results, metric="map@1"):
    """
    Возвращает средние кривые обучения по фолдам для 'train' и 'valid'.

    Parameters:
    - all_evals_results: список словарей, каждый из которых имеет структуру:
                         {'train': {'metric': [...]} , 'valid': {'metric': [...]}}
    - metric: название метрики, например 'map@1'

    Returns:
    - dict с ключами 'train' и 'valid', каждый из которых содержит усреднённую кривую
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
