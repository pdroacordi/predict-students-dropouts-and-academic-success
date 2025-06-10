"""
Classe base para todos os modelos
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any
import joblib
from pathlib import Path


class BaseModel(ABC):
    """Classe base abstrata para modelos"""

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.best_params = None

    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """Retorna o grid de hiperparâmetros para busca"""
        pass

    @abstractmethod
    def create_model(self, **params) -> Any:
        """Cria uma instância do modelo com os parâmetros especificados"""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray, **params):
        """Treina o modelo"""
        self.model = self.create_model(**params)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Faz predições"""
        if self.model is None:
            raise ValueError("Modelo não foi treinado")
        return self.model.predict(X)

    def save(self, path: Path):
        """Salva o modelo"""
        joblib.dump(self.model, path / f"{self.name}_model.pkl")

    def load(self, path: Path):
        """Carrega o modelo"""
        self.model = joblib.load(path / f"{self.name}_model.pkl")