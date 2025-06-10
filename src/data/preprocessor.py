"""
Módulo para pré-processamento dos dados
"""
from typing import Tuple, Any

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Classe para pré-processamento dos dados"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='mean')

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Ajusta os transformadores aos dados"""
        logger.info("Ajustando pré-processadores")
        self.imputer.fit(X)
        X_imputed = self.imputer.transform(X)
        self.scaler.fit(X_imputed)
        self.label_encoder.fit(y)

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> tuple[Any, Any] | tuple[Any, None]:
        """Transforma os dados"""
        logger.info("Aplicando transformações")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)

        if y is not None:
            y_encoded = self.label_encoder.transform(y)
            return X_scaled, y_encoded
        return X_scaled, None

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ajusta e transforma os dados"""
        self.fit(X, y)
        return self.transform(X, y)