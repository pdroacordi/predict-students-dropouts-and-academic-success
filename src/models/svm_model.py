"""
Implementação do modelo SVM
"""
from sklearn.svm import SVC
from .base_model import BaseModel
from typing import Dict, Any


class SVMModel(BaseModel):
    """Modelo Support Vector Machine"""

    def __init__(self):
        super().__init__("svm")

    def get_param_grid(self) -> Dict[str, Any]:
        """Retorna o grid de hiperparâmetros para SVM"""
        return {
            'C': [0.1, 1, 10, 100],
            'kernel': ['poly', 'rbf'],
            'degree': [2, 3, 4],  # apenas para kernel poly
            'gamma': ['scale', 'auto']
        }

    def create_model(self, **params) -> SVC:
        """Cria instância do SVM"""
        # Remove degree se kernel não for poly
        if params.get('kernel') != 'poly' and 'degree' in params:
            params.pop('degree')
        return SVC(random_state=42, **params)