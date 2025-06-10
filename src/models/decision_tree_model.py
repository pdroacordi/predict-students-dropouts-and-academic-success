"""
Implementação do modelo Árvore de Decisão
"""
from sklearn.tree import DecisionTreeClassifier
from .base_model import BaseModel
from typing import Dict, Any


class DecisionTreeModel(BaseModel):
    """Modelo Árvore de Decisão"""

    def __init__(self):
        super().__init__("decision_tree")

    def get_param_grid(self) -> Dict[str, Any]:
        """Retorna o grid de hiperparâmetros para Árvore de Decisão"""
        return {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def create_model(self, **params) -> DecisionTreeClassifier:
        """Cria instância da Árvore de Decisão"""
        return DecisionTreeClassifier(random_state=42, **params)