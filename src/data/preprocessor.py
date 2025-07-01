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
        self._is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Ajusta os transformadores APENAS aos dados de treino"""
        logger.info("Ajustando pré-processadores APENAS nos dados de treino")

        # Primeiro imputa valores faltantes
        self.imputer.fit(X_train)
        X_train_imputed = self.imputer.transform(X_train)

        # Depois normaliza com os dados já imputados
        self.scaler.fit(X_train_imputed)

        # Ajusta encoder de labels
        self.label_encoder.fit(y_train)

        self._is_fitted = True
        logger.info("Pré-processadores ajustados com sucesso")

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """Transforma apenas as features (sem labels)"""
        if not self._is_fitted:
            raise ValueError("Transformadores não foram ajustados. Chame fit() primeiro.")

        logger.info("Aplicando transformações nas features")
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def transform_labels(self, y: np.ndarray) -> np.ndarray:
        """Transforma apenas os labels"""
        if not self._is_fitted:
            raise ValueError("Transformadores não foram ajustados. Chame fit() primeiro.")

        return self.label_encoder.transform(y)

    def transform(self, X: np.ndarray, y: np.ndarray = None) -> tuple[Any, Any] | tuple[Any, None]:
        """Transforma os dados usando transformadores já ajustados"""
        X_transformed = self.transform_features(X)

        if y is not None:
            y_transformed = self.transform_labels(y)
            return X_transformed, y_transformed
        return X_transformed, None

    def fit_transform_train(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ajusta nos dados de treino E transforma os dados de treino"""
        self.fit(X_train, y_train)
        return self.transform(X_train, y_train)

    def inverse_transform_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Reverte a transformação dos labels"""
        return self.label_encoder.inverse_transform(y_encoded)