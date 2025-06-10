"""
Implementação do modelo KNN
"""
from sklearn.neighbors import KNeighborsClassifier
from .base_model import BaseModel
from typing import Dict, Any


class KNNModel(BaseModel):
    """Modelo K-Nearest Neighbors"""

    def __init__(self):
        super().__init__("knn")

    def get_param_grid(self) -> Dict[str, Any]:
        """Retorna o grid de hiperparâmetros para KNN"""
        return {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance']
        }

    def create_model(self, **params) -> KNeighborsClassifier:
        """Cria instância do KNN"""
        return KNeighborsClassifier(**params)