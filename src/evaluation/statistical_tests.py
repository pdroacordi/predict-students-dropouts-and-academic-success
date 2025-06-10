"""
Módulo para testes estatísticos
"""
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class StatisticalTester:
    """Classe para realizar testes estatísticos"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    @staticmethod
    def kruskal_wallis_test(results: Dict[str, List[float]]) -> Tuple[float, float]:
        """
        Teste de Kruskal-Wallis para verificar se há diferença entre os grupos
        """
        logger.info("Realizando teste de Kruskal-Wallis")

        # Prepara dados para o teste
        groups = list(results.values())

        # Realiza o teste
        statistic, p_value = stats.kruskal(*groups)

        logger.info(f"Kruskal-Wallis - Estatística: {statistic:.4f}, p-valor: {p_value:.4f}")

        return statistic, p_value

    @staticmethod
    def mann_whitney_tests(results: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Testes de Mann-Whitney para comparações par a par
        """
        logger.info("Realizando testes de Mann-Whitney")

        model_names = list(results.keys())
        n_models = len(model_names)

        # Matriz para armazenar p-valores
        p_values = np.zeros((n_models, n_models))

        # Realiza comparações par a par
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1, model2 = model_names[i], model_names[j]

                # Teste de Mann-Whitney
                statistic, p_value = stats.mannwhitneyu(
                    results[model1],
                    results[model2],
                    alternative='two-sided'
                )

                p_values[i, j] = p_value
                p_values[j, i] = p_value

                logger.info(f"{model1} vs {model2}: p-valor = {p_value:.4f}")

        # Cria DataFrame com resultados
        df_pvalues = pd.DataFrame(p_values,
                                  index=model_names,
                                  columns=model_names)

        return df_pvalues

    def apply_bonferroni_correction(self, p_values: pd.DataFrame) -> bool:
        """
        Aplica correção de Bonferroni para múltiplas comparações
        """
        n_comparisons = (len(p_values) * (len(p_values) - 1)) / 2
        adjusted_alpha = self.alpha / n_comparisons

        logger.info(f"Alpha ajustado (Bonferroni): {adjusted_alpha:.4f}")

        return p_values < adjusted_alpha