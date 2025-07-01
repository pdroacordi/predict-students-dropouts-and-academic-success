"""
Módulo para validação cruzada e avaliação repetida
"""
import logging
from typing import Dict, Any, Tuple, List

import numpy as np
from tqdm import tqdm

from .metrics import MetricsCalculator
from ..models.base_model import BaseModel
from ..data.splitter import DataSplitter
from ..data.preprocessor import DataPreprocessor
from ..tuning.hyperparameter_tuning import HyperparameterTuner

logger = logging.getLogger(__name__)


class CrossValidator:
    """Classe para realizar validação cruzada repetida com divisões INDEPENDENTES"""

    def __init__(self, n_iterations: int = 10, random_state: int = 42,
                 train_size: float = 0.5, val_size: float = 0.25, test_size: float = 0.25):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.metrics_calculator = MetricsCalculator()

    def evaluate_model(self, model: BaseModel, X_original: np.ndarray, y_original: np.ndarray) -> Tuple[List[float], List[Dict[str, Any]], List[Dict]]:
        """
        Avalia modelo com 10 DIVISÕES INDEPENDENTES dos dados

        Para cada iteração:
        1. Nova divisão aleatória treino/val/teste
        2. Ajusta pré-processamento APENAS no treino
        3. Aplica transformações em val/teste
        4. Faz tuning de hiperparâmetros com treino+val
        5. Avalia modelo final no teste
        """
        logger.info(f"Avaliando {model.name} com {self.n_iterations} divisões independentes")

        accuracies = []
        all_metrics = []
        all_best_params = []

        for i in tqdm(range(self.n_iterations), desc=f"Avaliando {model.name}"):
            logger.info(f"Iteração {i+1}/{self.n_iterations} - {model.name}")

            # 1. NOVA divisão aleatória para cada iteração
            splitter = DataSplitter(
                train_size=self.train_size,
                val_size=self.val_size,
                test_size=self.test_size,
                random_state=self.random_state + i  # Seed diferente para cada iteração
            )

            splits = splitter.split(X_original, y_original)
            X_train, y_train = splits['train']
            X_val, y_val = splits['validation']
            X_test, y_test = splits['test']

            logger.info(f"Nova divisão - Treino: {len(X_train)}, Val: {len(X_val)}, Teste: {len(X_test)}")

            # 2. Pré-processamento SEM data leakage
            preprocessor = DataPreprocessor()

            # Ajusta APENAS no treino
            X_train_processed, y_train_processed = preprocessor.fit_transform_train(X_train, y_train)

            # Aplica transformações (sem ajustar) em validação e teste
            X_val_processed, y_val_processed = preprocessor.transform(X_val, y_val)
            X_test_processed, y_test_processed = preprocessor.transform(X_test, y_test)

            logger.info("Pré-processamento aplicado sem data leakage")

            # 3. Tuning de hiperparâmetros
            tuner = HyperparameterTuner(cv_folds=5, scoring='accuracy')
            best_params, best_score = tuner.tune_model(
                model, X_train_processed, y_train_processed,
                X_val_processed, y_val_processed
            )

            all_best_params.append(best_params)
            logger.info(f"Melhores params iteração {i+1}: {best_params}")

            # 4. Treina modelo final com melhores parâmetros
            model.fit(X_train_processed, y_train_processed, **best_params)

            # 5. Avalia APENAS no conjunto de teste (nunca visto antes)
            y_pred = model.predict(X_test_processed)

            # 6. Calcula métricas
            metrics = self.metrics_calculator.calculate_metrics(y_test_processed, y_pred)
            accuracies.append(metrics['accuracy'])
            all_metrics.append(metrics)

            logger.info(f"Acurácia iteração {i+1}: {metrics['accuracy']:.4f}")

        logger.info(f"Avaliação completa - {model.name}")
        logger.info(f"Acurácia média: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

        return accuracies, all_metrics, all_best_params