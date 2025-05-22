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


SEED = 2007
N_SPLITS = 5

# Загрузка данных
data = read_merge_data('./data_train_crossdock_rmsd_ccf.csv')

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

def get_reduce_Xy(df, svd_model, target_variance = 0.95):
    X = extract_X(df)
    X = svd_model.transform(X)
    y = df['labels'].values
    return X, y

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


def plot_combined_learning_curves(all_evals_results, metric='map@1'):
    """
    all_evals_results: список словарей с результатами обучения (по фолдам)
                       каждый элемент соответствует одному фолду
                       и имеет структуру как evals_result из XGBRanker.fit()
    metric: название метрики, которую нужно визуализировать (например, 'map@1')
    """
    print(all_evals_results)
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

    # Строим график
    plt.figure(figsize=(10, 5))
    for c in trimmed_curves:
        plt.plot(range(1, max_len+1), c, color='gray', alpha=0.5)

    median_curve = np.median(trimmed_curves, axis=0)
    plt.plot(range(1, max_len+1), median_curve, color='red', linewidth=2, label='Медианная кривая')

    plt.title(f'Кривая обучения XGBRanker (метрика: {metric})')
    plt.xlabel('Boosting итерации')
    plt.ylabel(metric)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


################################################################################
################################################################################
# ---- Data Preparation ----
X_raw = extract_X(data)
_, SVD_model = reduce_dim(X_raw)

# ---- LOLO Evaluation ----
print("\n=== LOLO Evaluation ===")
lolo_accuracies = []
per_ligand_scores = []  # list of dicts for later plotting
unique_ligands = data['init_ligand_file'].unique()
evals_results = []

for test_ligand in tqdm(unique_ligands[:3], desc="LOLO Evaluation"):
    #print(test_ligand)
    df_train = data[data['init_ligand_file'] != test_ligand].copy()
    df_test = data[data['init_ligand_file'] == test_ligand].copy()

    X_train, y_train = get_reduce_Xy(df_train, SVD_model)
    X_test, y_test = get_reduce_Xy(df_test, SVD_model)

    group_train = df_train.groupby('init_ligand_file').size().tolist()
    group_test = [len(df_test)]

    if 1 not in y_test:
        print(f'Skip data for {test_ligand} as it does not have any \
        native-like pose. Nothing to range')
        print()
    else:
        model = XGBRanker(
            objective='rank:ndcg',  # накладывает большой штраф на label==1, если он не в топе
            eval_metric=['ndcg@1', 'map@1', 'map'], # следи за top-1 на валидации
            learning_rate=0.05,  # Trying smaller learning rate
            max_depth=4,         # Deeper trees might help
            n_estimators=200,    # More trees might improve accuracy
            subsample=0.8,       # Subsampling to prevent overfitting
            colsample_bytree=0.8, # Randomizing features used by trees
            random_state=SEED
        )
        #print('data.shape', data.shape)
        #print('X_train.shape', X_train.shape)
        #print('y_test.shape', y_test.shape)
        #print('the number of native-like poses in y_test =', len(y_test[y_test==1]))
        model.fit(
                    X_train, y_train,
                    group=group_train,
                    eval_set=[(X_test, y_test)],
                    eval_group=[group_test],
                    verbose=False
                    )
        evals_results.append(list(model.evals_result().values())[0])

        # ---- Data Retrieval for Threshold Choosing Plot ----
        df_test['score'] = model.predict(X_test)
        df_test['true_label'] = y_test
        top_ranked = df_test.sort_values(by='score', ascending=False).groupby('init_ligand_file').head(1)
        top_score = top_ranked['score'].iloc[0]
        is_native_like = top_ranked['true_label'].iloc[0]
        any_native_like = df_test['true_label'].sum() > 0  # check if ligand has any good pose
        n_cl = top_ranked['n_cluster'].iloc[0]
        per_ligand_scores.append({
            'ligand': test_ligand,
            'top_score': top_score,
            'top_rank_is_native': is_native_like,
            'has_native_like_pose': any_native_like,
    	'n_cluster': n_cl
        })
        # plot_learning_curve(evals_results, fold) # если нужно отследить обучение внутри одного прогона
        #lolo_accuracies.append(top_ranked['true_label'].iloc[0])


#print('lolo_accuracies', lolo_accuracies)
#lolo_mean = np.mean(lolo_accuracies)
#print(f"\nLOLO Top-1 Mean Accuracy for LOLO: {lolo_mean:.3f}")
plot_combined_learning_curves(evals_results)


# ---- Plot the Results ----
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
