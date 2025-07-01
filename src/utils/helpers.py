"""
Funções auxiliares
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def save_results(results: Dict[str, Any], filepath: Path):
    """Salva resultados em arquivo JSON"""

    # Converte arrays numpy para listas
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        elif isinstance(obj, pd.Series):
            return obj.to_list()
        return obj

    converted_results = convert_numpy(results)

    with open(filepath, 'w') as f:
        json.dump(converted_results, f, indent=4)


def load_results(filepath: Path) -> Dict[str, Any]:
    """Carrega resultados de arquivo JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confiança"""
    import scipy.stats as stats

    n = len(data)
    mean = np.mean(data)
    stderr = stats.sem(data)

    # Graus de liberdade
    df = n - 1

    # Valor crítico t
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    # Margem de erro
    margin = t_critical * stderr

    return (mean - margin, mean + margin)


def format_time(seconds: float) -> str:
    """Formata tempo em formato legível"""
    if seconds < 60:
        return f"{seconds:.2f} segundos"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f} minutos"
    else:
        hours = seconds / 3600
        return f"{hours:.2f} horas"