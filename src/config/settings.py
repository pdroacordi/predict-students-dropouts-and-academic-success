"""
Configurações globais do projeto
"""
import os
from pathlib import Path

# Caminhos
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Configurações de divisão dos dados
TRAIN_SIZE = 0.5
VAL_SIZE = 0.25
TEST_SIZE = 0.25
RANDOM_STATE = 42
N_ITERATIONS = 10

# Hiperparâmetros para busca
HYPERPARAMETERS = {
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15],
        'weights': ['uniform', 'distance']
    },
    'decision_tree': {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 5, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'svm': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['poly', 'rbf'],
        'degree': [2, 3, 4],  # apenas para kernel poly
        'gamma': ['scale', 'auto']
    },
    'mlp': {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [200, 500, 1000]
    }
}

# Configurações de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"