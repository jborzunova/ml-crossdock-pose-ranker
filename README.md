# ml-crossdock-pose-ranker

**Machine Learning model (XGBRanker) for cross-docking pose ranking.**  
This project ranks docking poses so that the top-1 pose is native-like, improving accuracy in molecular docking tasks.

> **Note:** This project is still under active development and intended to be published in a scientific article.
>
> - The figures shown below are **preliminary** and may change.
> - The dataset features are still being refined and extended.
> - Code, documentation, and results will be updated regularly as the project evolves.

## Features

- Utilizes XGBRanker from XGBoost for ranking molecular docking poses.
- Focused on cross-docking scenarios.
- Includes training, evaluation, and visualization scripts.
- Configurable parameters via `parameters.py`.
- Easy integration with your dataset.

## Repository Structure

`````
├── app/ # Application code
├── images/
│   ├── all_valid_learning_curves.png
│   └── topk_learning_curves.png
|   └── best_model_learning_curves.png
├── data/
|   ├── raw
|       ├── data_cross_ref_natives_ccf_rmsd.csv # Dataset in CSV format
|   └── learning_curves  # values of plots
|   └── processed  # here datasets for every cross-validation loop will be saved
├── optimize_params.py  # Main script for Optuna search
├── run_best_model.py  # run this script after Optuna search
├── parameters.py # Configuration and parameters for the model
├── README.md # This documentation file
`````

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:jborzunova/ml-crossdock-pose-ranker.git
   cd ml-crossdock-pose-ranker

    (Optional) Create and activate a virtual environment. Install dependencies listed in requirements.txt

Usage:

Run training or inference with:

	python3 main.py

Customize parameters in parameters.py.

## Results

### Top-K Learning Curves (Train & Validation)
![Top-K Learning Curves](images/topk_learning_curves.png)

This plot illustrates training and validation curves for the **top-K models** (ranked by validation performance).  
It provides insight into model convergence and stability across the most successful configurations.

### Best Model Learning Curves (Train & Validation)
![Best Model Learning Curves (Train & Validation)](images/best_model_learning_curves.png)

This plot shows the training and validation learning curves for the single best model, selected based on the highest validation performance across all configurations.
It reflects how well the best configuration generalized during cross-validation, and can be used to assess overfitting or underfitting tendencies.

## Dataset

The dataset used in this project currently consists of **CCF fingerprints** representing molecular features relevant for cross-docking pose ranking.  

These fingerprints capture key chemical characteristics and serve as input features for the XGBRanker model.  

**Note:** Feature engineering is ongoing, and additional descriptors may be added in future updates.

License: [No license yet]

Contact:

Julia Borzunova — [j.n.borzunova@yandex.ru]
GitHub: https://github.com/jborzunova
