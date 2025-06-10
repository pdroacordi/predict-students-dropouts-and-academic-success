"""
Módulo para tuning de hiperparâmetros
"""
import numpy as np
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, Tuple
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Classe para realizar busca de hiperparâmetros"""

    def __init__(self, cv_folds: int = 5, scoring: str = 'accuracy', n_jobs: int = -1):
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.results = {}

    def tune_model(self, model, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Dict[str, Any], float]:
        """Realiza busca de hiperparâmetros para um modelo"""
        logger.info(f"Iniciando tuning para {model.name}")

        param_grid = model.get_param_grid()
        base_model = model.create_model()

        # GridSearch com validação cruzada
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            verbose=1
        )

        # Combina treino e validação para o tuning
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])

        grid_search.fit(X_combined, y_combined)

        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        logger.info(f"Melhores parâmetros para {model.name}: {best_params}")
        logger.info(f"Melhor score: {best_score:.4f}")

        # Salva resultados detalhados
        self.results[model.name] = {
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_
        }

        return best_params, best_score

    def save_results(self, path: Path):
        """Salva os resultados do tuning"""
        results_path = path / "hyperparameter_results.json"

        # Converte arrays numpy para listas
        results_to_save = {}
        for model_name, result in self.results.items():
            results_to_save[model_name] = {
                'best_params': result['best_params'],
                'best_score': float(result['best_score'])
            }

        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=4)