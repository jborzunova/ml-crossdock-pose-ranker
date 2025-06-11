# ml-crossdock-pose-ranker

**Machine Learning model (XGBRanker) for cross-docking pose ranking.**  
This project ranks docking poses so that the top-1 pose is native-like, improving accuracy in molecular docking tasks.

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
├── data_crossdock_rmsd_ccf.csv # Dataset in CSV format
├── main.py # Main script for training or inference
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

### Validation Learning Curves from Optuna Trials
![All Validation Learning Curves](images/all_valid_learning_curves.png)

This figure shows all validation learning curves obtained during hyperparameter optimization with **Optuna**.  
The validation was performed using the **Leave-One-Ligand-Out (LOLO)** strategy to ensure robust generalization across ligands.  
The **best-performing trial** (as selected by Optuna) is highlighted in **red**.

### Top-K Learning Curves (Train & Validation)
![Top-K Learning Curves](images/topk_learning_curves.png)

This plot illustrates training and validation curves for the **top-K models** (ranked by validation performance).  
It provides insight into model convergence and stability across the most successful configurations.

License: [No license yet]

Contact:

Julia Borzunova — [j.n.borzunova@yandex.ru]
GitHub: https://github.com/jborzunova
