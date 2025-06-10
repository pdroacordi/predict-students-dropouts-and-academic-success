"""
Módulo para criação de visualizações
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List
from pathlib import Path


class Plotter:
    """Classe para criar visualizações"""

    def __init__(self, style: str = 'seaborn'):
        plt.style.use(style)
        sns.set_palette("husl")

    def plot_accuracy_comparison(self, results: Dict[str, List[float]],
                                 save_path: Path = None):
        """Cria boxplot comparando acurácias dos modelos"""
        fig, ax = plt.subplots(figsize=(10, 6))

        # Prepara dados
        data = []
        labels = []
        for model_name, accuracies in results.items():
            data.append(accuracies)
            labels.append(model_name)

        # Cria boxplot
        bp = ax.boxplot(data, labels=labels, patch_artist=True)

        # Customiza cores
        colors = sns.color_palette("husl", len(data))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xlabel('Modelo')
        ax.set_ylabel('Acurácia')
        ax.set_title('Comparação de Acurácia entre Modelos')
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_statistical_heatmap(self, p_values: pd.DataFrame, save_path: Path = None):
        """Cria heatmap dos p-valores dos testes estatísticos"""
        fig, ax = plt.subplots(figsize=(8, 6))

        # Máscara para triângulo superior
        mask = np.triu(np.ones_like(p_values), k=1)

        # Cria heatmap
        sns.heatmap(p_values,
                    mask=mask,
                    annot=True,
                    fmt='.4f',
                    cmap='RdYlBu_r',
                    center=0.05,
                    square=True,
                    linewidths=1,
                    cbar_kws={"shrink": .8})

        ax.set_title('P-valores dos Testes de Mann-Whitney')

        if save_path:
            plt.savefig(save_path / 'statistical_tests_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_metrics_radar(self, metrics: Dict[str, Dict[str, float]],
                           save_path: Path = None):
        """Cria gráfico radar comparando múltiplas métricas"""
        # Prepara dados
        categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        models = list(metrics.keys())

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

        # Ângulos para cada métrica
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        # Plot para cada modelo
        for model in models:
            values = [
                metrics[model]['accuracy'],
                metrics[model]['precision'],
                metrics[model]['recall'],
                metrics[model]['f1_score']
            ]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=model)
            ax.fill(angles, values, alpha=0.25)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('Comparação de Métricas entre Modelos', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        if save_path:
            plt.savefig(save_path / 'metrics_radar.png', dpi=300, bbox_inches='tight')
        plt.show()