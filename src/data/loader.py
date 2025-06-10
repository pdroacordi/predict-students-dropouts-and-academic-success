"""
Módulo para carregamento dos dados
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Classe para carregar e verificar os dados"""

    def __init__(self, data_path: Path):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados do arquivo CSV"""
        try:
            logger.info(f"Carregando dados de {self.data_path}")
            df = pd.read_csv(self.data_path)
            logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            raise

    def get_features_and_target(self, df: pd.DataFrame,
                                target_column: str = 'Target') -> Tuple[np.ndarray, np.ndarray]:
        """Separa features e target"""
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        return X, y