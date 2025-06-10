"""
Módulo para divisão dos dados
"""
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """Classe para dividir os dados em treino, validação e teste"""

    def __init__(self, train_size: float = 0.5,
                 val_size: float = 0.25,
                 test_size: float = 0.25,
                 random_state: int = 42):
        assert train_size + val_size + test_size == 1.0
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state

    def split(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Divide os dados em treino, validação e teste"""
        logger.info(f"Dividindo dados: {self.train_size:.0%} treino, "
                    f"{self.val_size:.0%} validação, {self.test_size:.0%} teste")

        # Primeiro separa treino do resto
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y,
            test_size=(1 - self.train_size),
            stratify=y,
            random_state=self.random_state
        )

        # Depois separa validação e teste
        val_size_adjusted = self.val_size / (self.val_size + self.test_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size_adjusted),
            stratify=y_temp,
            random_state=self.random_state
        )

        logger.info(f"Tamanhos dos conjuntos - Treino: {len(X_train)}, "
                    f"Validação: {len(X_val)}, Teste: {len(X_test)}")

        return {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }