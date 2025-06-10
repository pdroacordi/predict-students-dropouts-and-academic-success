"""
Implementação do modelo MLP
"""
from sklearn.neural_network import MLPClassifier
from .base_model import BaseModel
from typing import Dict, Any


class MLPModel(BaseModel):
    """Modelo Multi-Layer Perceptron"""

    def __init__(self):
        super().__init__("mlp")

    def get_param_grid(self) -> Dict[str, Any]:
        """Retorna o grid de hiperparâmetros para MLP"""
        return {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['identity', 'logistic', 'tanh', 'relu'],
            'learning_rate_init': [0.001, 0.01, 0.1],
            'max_iter': [200, 500, 1000]
        }

    def create_model(self, **params) -> MLPClassifier:
        """Cria instância do MLP"""
        return MLPClassifier(random_state=42, early_stopping=True, **params)