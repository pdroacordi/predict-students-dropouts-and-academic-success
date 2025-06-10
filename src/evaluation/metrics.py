"""
Módulo para cálculo de métricas
"""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from typing import Dict, Any
import pandas as pd


class MetricsCalculator:
    """Classe para calcular métricas de avaliação"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcula várias métricas de classificação"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }

        return metrics

    @staticmethod
    def create_metrics_dataframe(results: Dict[str, list]) -> [pd.DataFrame, pd.DataFrame]:
        """Cria DataFrame com resultados das métricas"""
        df = pd.DataFrame(results)

        # Adiciona estatísticas descritivas
        df_stats = df.describe().T
        df_stats['cv'] = df.std() / df.mean()  # Coeficiente de variação

        return df, df_stats