"""
Módulo para geração de relatórios
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


class ReportGenerator:
    """Classe para gerar relatórios detalhados"""

    def __init__(self, results_dir: Path):
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def generate_html_report(self,
                             model_results: Dict[str, Any],
                             statistical_results: Dict[str, Any],
                             best_model: str):
        """Gera relatório em formato HTML"""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Comparação de Algoritmos ML</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .highlight {{ background-color: #e6f3ff; font-weight: bold; }}
                .section {{ margin: 30px 0; }}
                .metrics-table {{ width: auto; }}
                .statistical-significance {{ color: #d9534f; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Comparação de Algoritmos de Machine Learning</h1>
            <p><strong>Data:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>

            <div class="section">
                <h2>1. Resumo Executivo</h2>
                <p>Este relatório apresenta a comparação de desempenho entre 4 algoritmos de Machine Learning:</p>
                <ul>
                    <li>K-Nearest Neighbors (KNN)</li>
                    <li>Árvore de Decisão</li>
                    <li>Support Vector Machine (SVM)</li>
                    <li>Multi-Layer Perceptron (MLP)</li>
                </ul>
                <p><strong>Melhor modelo identificado:</strong> <span class="highlight">{best_model}</span></p>
            </div>

            <div class="section">
                <h2>2. Melhores Hiperparâmetros</h2>
                {self._create_hyperparameters_table(model_results)}
            </div>

            <div class="section">
                <h2>3. Métricas de Desempenho</h2>
                {self._create_metrics_table(model_results)}
            </div>

            <div class="section">
                <h2>4. Análise Estatística</h2>
                <h3>4.1 Teste de Kruskal-Wallis</h3>
                <p>Estatística: {statistical_results['kruskal_wallis']['statistic']:.4f}</p>
                <p>P-valor: {statistical_results['kruskal_wallis']['p_value']:.4f}</p>
                <p><strong>Conclusão:</strong> {self._interpret_kruskal_wallis(statistical_results['kruskal_wallis']['p_value'])}</p>

                <h3>4.2 Testes de Mann-Whitney (comparações par a par)</h3>
                {self._create_mann_whitney_table(statistical_results['mann_whitney'])}
            </div>

            <div class="section">
                <h2>5. Conclusões</h2>
                {self._generate_conclusions(model_results, statistical_results, best_model)}
            </div>
        </body>
        </html>
        """

        # Salva o relatório
        report_path = self.results_dir / f"relatorio_completo_{self.timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return report_path

    def _create_hyperparameters_table(self, model_results: Dict) -> str:
        """Cria tabela HTML com hiperparâmetros"""
        table = "<table><tr><th>Modelo</th><th>Hiperparâmetros</th></tr>"

        for model, results in model_results.items():
            params = results['best_params']
            params_str = "<br>".join([f"{k}: {v}" for k, v in params.items()])
            table += f"<tr><td>{model}</td><td>{params_str}</td></tr>"

        table += "</table>"
        return table

    def _create_metrics_table(self, model_results: Dict) -> str:
        """Cria tabela HTML com métricas"""
        table = """
        <table class="metrics-table">
        <tr>
            <th>Modelo</th>
            <th>Acurácia Média</th>
            <th>Desvio Padrão</th>
            <th>Mín</th>
            <th>Máx</th>
            <th>Precisão</th>
            <th>Recall</th>
            <th>F1-Score</th>
        </tr>
        """

        for model, results in model_results.items():
            acc = results['accuracies']
            metrics = results['final_metrics']

            row_class = 'highlight' if results.get('is_best', False) else ''

            table += f"""
            <tr class="{row_class}">
                <td>{model}</td>
                <td>{np.mean(acc):.4f}</td>
                <td>{np.std(acc):.4f}</td>
                <td>{np.min(acc):.4f}</td>
                <td>{np.max(acc):.4f}</td>
                <td>{metrics['precision']:.4f}</td>
                <td>{metrics['recall']:.4f}</td>
                <td>{metrics['f1_score']:.4f}</td>
            </tr>
            """

        table += "</table>"
        return table

    def _interpret_kruskal_wallis(self, p_value: float) -> str:
        """Interpreta resultado do teste de Kruskal-Wallis"""
        if p_value < 0.05:
            return "Há diferença estatisticamente significativa entre os modelos (p < 0.05)"
        else:
            return "Não há diferença estatisticamente significativa entre os modelos (p ≥ 0.05)"

    def _create_mann_whitney_table(self, mann_whitney_results: pd.DataFrame) -> str:
        """Cria tabela com resultados de Mann-Whitney"""
        table = "<table><tr><th>Comparação</th><th>P-valor</th><th>Significância</th></tr>"

        models = mann_whitney_results.index
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                p_value = mann_whitney_results.iloc[i, j]
                sig = "Sim" if p_value < 0.05 else "Não"
                sig_class = "statistical-significance" if p_value < 0.05 else ""

                table += f"""
                <tr>
                    <td>{models[i]} vs {models[j]}</td>
                    <td>{p_value:.4f}</td>
                    <td class="{sig_class}">{sig}</td>
                </tr>
                """

        table += "</table>"
        return table

    def _generate_conclusions(self, model_results: Dict,
                              statistical_results: Dict,
                              best_model: str) -> str:
        """Gera conclusões do relatório"""
        conclusions = f"""
        <p>Com base nos resultados obtidos, podemos concluir que:</p>
        <ul>
            <li>O modelo <strong>{best_model}</strong> apresentou o melhor desempenho geral, 
                com acurácia média de {np.mean(model_results[best_model]['accuracies']):.4f}</li>
        """

        # Adiciona análise estatística
        if statistical_results['kruskal_wallis']['p_value'] < 0.05:
            conclusions += """
            <li>O teste de Kruskal-Wallis indicou diferenças significativas entre os modelos</li>
            <li>As comparações par a par (Mann-Whitney) revelaram:</li>
            <ul>
            """

            # Analisa comparações significativas
            mann_whitney = statistical_results['mann_whitney']
            for i in range(len(mann_whitney)):
                for j in range(i + 1, len(mann_whitney)):
                    if mann_whitney.iloc[i, j] < 0.05:
                        model1, model2 = mann_whitney.index[i], mann_whitney.index[j]
                        conclusions += f"<li>Diferença significativa entre {model1} e {model2}</li>"

            conclusions += "</ul>"
        else:
            conclusions += """
            <li>O teste de Kruskal-Wallis não indicou diferenças significativas entre os modelos</li>
            """

        conclusions += """
        </ul>
        <p>Recomenda-se o uso do modelo {} para esta tarefa de classificação.</p>
        """.format(best_model)

        return conclusions

    def save_summary_json(self, results: Dict[str, Any]):
        """Salva resumo dos resultados em JSON"""
        summary_path = self.results_dir / f"summary_{self.timestamp}.json"

        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=4, default=str)

        return summary_path