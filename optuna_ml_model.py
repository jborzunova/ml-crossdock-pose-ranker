# conda activate ml
import pandas as pd
import numpy as np
from xgboost import XGBRanker
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
from read_merge_data import *
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
import optuna


SEED = 2007
N_SPLITS = 5
METRIC = 'map@1'

def reduce_dim(X, target_variance = 0.95):
    # ---- Уменьшение размерности данных ----
    # --- Автовыбор n_components по explained variance ---
    print('Reducing Dimensions of Data ...')
    max_components = min(X.shape[0] - 1, X.shape[1])
    # Пробуем много компонент и проверяем накопленную дисперсию
    svd_full = TruncatedSVD(n_components=max_components, random_state=42)
    svd_full.fit(X)
    explained_variance = np.cumsum(svd_full.explained_variance_ratio_)
    n_components_optimal = np.argmax(explained_variance >= target_variance) + 1
    print(f"Выбрано {n_components_optimal} компонент для сохранения {target_variance:.0%} дисперсии.")
    # Теперь применим SVD с оптимальным количеством компонент
    svd = TruncatedSVD(n_components=n_components_optimal, random_state=SEED)
    X_reduced = svd.fit_transform(X)
    print('the new shape of features is', X_reduced.shape)
    return X_reduced, svd


def prepare_XGB_data(df, svd_model, target_variance=0.95):
    # Cортировка необходима для модели ranker!
    # (лист group говорят модели размер последовательных групп)
    df = df.sort_values(by='init_ligand_file').reset_index(drop=True)
    #print(df.loc[:150, 'init_ligand_file'])
    y = df['labels'].values
    X_raw = extract_X(df)
    X_reduced = svd_model.transform(X_raw)
    #X_df = pd.DataFrame(X_reduced, index=df.index)
    #print(type(X_reduced))
    # Считаем размер каждой группы
    group = df.groupby('init_ligand_file').size().tolist()
    return X_reduced, y, group


def extract_X(df):
    return df.drop(columns=['Unnamed: 0', 'labels', 'init_ligand_file', 'n_cluster',
					'fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4'])

def plot_learning_curve(evals_results, fold):
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


################################################################################
################################################################################
# ---- Data Preparation ----
data = read_merge_data('./data_train_crossdock_rmsd_ccf.csv')  # Загрузка данных
X_raw = extract_X(data)
_, SVD_model = reduce_dim(X_raw)

# ---- Data Training and LOLO Evaluation ----
print("\n=== Optuna Parameters Optimization ===")
#per_ligand_scores = []  # list of dicts for later plotting
unique_ligands = data['init_ligand_file'].unique()
evals_results = []
learning_curves_by_trial = {}  # словарь для сохранения средних кривых обучения
# для разных параметров модели (которые оптимизируются в процессе работы Optuna)

def objective(trial):
    '''
    This function helps find the best parameters for ML model
    It trains the model and evaluate it on validation set with LOLO algorithm.
    Then all Learning Curves (validation on each ligand) are gathered to obtain
    a mean Learning Curve. The output of this function is the last point of it.
    Based on this value the Optuna will decide which parameters give the best
    result
    '''
    # ---- LOLO algorithm ----
    for test_ligand in tqdm(unique_ligands,
                            desc=f"LOLO Evaluation {trial.number}"):
        # ---- Prepare Data ----
        df_train = data[data['init_ligand_file'] != test_ligand].copy()
        df_test = data[data['init_ligand_file'] == test_ligand].copy()
        X_train, y_train, group_train = prepare_XGB_data(df_train, SVD_model)
        X_test, y_test, group_test = prepare_XGB_data(df_test, SVD_model)
        # ---- Validation only on data with at least one native like pose ----
        if 1 in y_test:  # Skip data if does not have any native-like pose. Nothing to rank
            params = {
                        'objective': 'rank:ndcg',
                        'eval_metric': METRIC,
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 10),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'gamma': trial.suggest_float('gamma', 0, 5),
                        'n_estimators': 1000, # максимальное число итераций доступно
                        'early_stopping_rounds': 200, # но если не происходит увеличения метрики в течение 30 шагов, остановись
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), # Это доля признаков (фичей), которая будет случайно отобрана для построения каждого дерева.
                        'random_state': SEED
                      }
            model = XGBRanker(**params)
            model.fit(
                        X_train, y_train,
                        group=group_train,
                        eval_set=[(X_test, y_test)],
                        eval_group=[group_test],
                        verbose=False
                        )
            evals_results.append(list(model.evals_result().values())[0])
            '''
            # ---- Data Retrieval for Threshold Choosing Plot ----
            df_test['score'] = model.predict(X_test)
            df_test['true_label'] = y_test
            top_ranked = df_test.sort_values(by='score', ascending=False).groupby('init_ligand_file').head(1)
            top_score = top_ranked['score'].iloc[0]
            is_native_like = top_ranked['true_label'].iloc[0]
            n_cl = top_ranked['n_cluster'].iloc[0]
            per_ligand_scores.append({
                'ligand': test_ligand,
                'top_score': top_score,
                'top_rank_is_native': is_native_like,
        	    'n_cluster': n_cl
            })
            '''
            # plot_learning_curve(evals_results, fold) # если нужно отследить обучение внутри одного прогона
    # ---- Get the Result of Cross-Validation for Optuna ----
    mean_curve = get_combined_learning_curve(evals_results)
    learning_curves_by_trial[trial.number] = mean_curve
    return mean_curve[-1]

# ---- Optimize Model Parameters ----
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
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
