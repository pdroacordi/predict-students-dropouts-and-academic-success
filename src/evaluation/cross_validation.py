"""
Módulo para validação cruzada e avaliação repetida
"""
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

from .metrics import MetricsCalculator
from ..models.base_model import BaseModel

logger = logging.getLogger(__name__)


class CrossValidator:
    """Classe para realizar validação cruzada repetida"""

    def __init__(self, n_iterations: int = 10, random_state: int = 42):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.metrics_calculator = MetricsCalculator()

    def evaluate_model(self, model: BaseModel, best_params: Dict,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Avalia modelo múltiplas vezes no conjunto de teste com bootstrap"""
        logger.info(f"Avaliando {model.name} por {self.n_iterations} iterações")

        accuracies = []
        all_metrics = []

        for i in tqdm(range(self.n_iterations), desc=f"Avaliando {model.name}"):
            # Bootstrap sampling do conjunto de treino para introduzir variabilidade
            X_train_boot, y_train_boot = resample(
                X_train, y_train,
                n_samples=len(X_train),
                random_state=self.random_state + i  # Seed diferente para cada iteração
            )

            # Treina modelo com dados bootstrapped
            model.fit(X_train_boot, y_train_boot, **best_params)

            # Prediz no conjunto de teste original
            y_pred = model.predict(X_test)

            # Calcula métricas
            metrics = self.metrics_calculator.calculate_metrics(y_test, y_pred)
            accuracies.append(metrics['accuracy'])
            all_metrics.append(metrics)

        return accuracies, all_metrics