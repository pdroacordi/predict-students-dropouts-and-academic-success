"""
Script principal para executar a comparação de algoritmos
"""
import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Adiciona src ao path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

# Imports dos módulos do projeto
from src.config.settings import *
from src.data.loader import DataLoader
from src.data.preprocessor import DataPreprocessor
from src.data.splitter import DataSplitter
from src.models.knn_model import KNNModel
from src.models.decision_tree_model import DecisionTreeModel
from src.models.svm_model import SVMModel
from src.models.mlp_model import MLPModel
from src.tuning.hyperparameter_tuning import HyperparameterTuner
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.statistical_tests import StatisticalTester
from src.evaluation.metrics import MetricsCalculator
from src.visualization.plots import Plotter
from src.visualization.report_generator import ReportGenerator
from src.utils.logger import setup_logger
from src.utils.helpers import save_results, format_time


def main():
    """Função principal"""

    # Configuração inicial
    start_time = time.time()

    # Cria diretórios necessários
    for dir_path in [PROCESSED_DATA_DIR,
                     RESULTS_DIR / "hyperparameters",
                     RESULTS_DIR / "predictions",
                     RESULTS_DIR / "metrics",
                     RESULTS_DIR / "plots"]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Configura logger
    logger = setup_logger("main", RESULTS_DIR)
    logger.info("=" * 80)
    logger.info("INICIANDO COMPARAÇÃO DE ALGORITMOS DE MACHINE LEARNING")
    logger.info("=" * 80)

    # 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
    logger.info("\n1. CARREGANDO E PREPARANDO DADOS")

    # Carrega dados
    data_loader = DataLoader(RAW_DATA_DIR / "students_dropout.csv")
    df = data_loader.load_data()
    X, y = data_loader.get_features_and_target(df)

    # Pré-processamento
    preprocessor = DataPreprocessor()
    X_processed, y_processed = preprocessor.fit_transform(X, y)

    # Divisão dos dados
    splitter = DataSplitter(
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    data_splits = splitter.split(X_processed, y_processed)

    X_train, y_train = data_splits['train']
    X_val, y_val = data_splits['validation']
    X_test, y_test = data_splits['test']

    # 2. DEFINIÇÃO DOS MODELOS
    logger.info("\n2. INICIALIZANDO MODELOS")
    models = {
        'KNN': KNNModel(),
        'Decision_Tree': DecisionTreeModel(),
        'SVM': SVMModel(),
        'MLP': MLPModel()
    }

    # 3. TUNING DE HIPERPARÂMETROS
    logger.info("\n3. REALIZANDO TUNING DE HIPERPARÂMETROS")
    tuner = HyperparameterTuner(cv_folds=5, scoring='accuracy')
    best_params = {}

    for model_name, model in models.items():
        logger.info(f"\nTuning {model_name}...")
        params, score = tuner.tune_model(model, X_train, y_train, X_val, y_val)
        best_params[model_name] = params

    # Salva melhores parâmetros
    tuner.save_results(RESULTS_DIR / "hyperparameters")

    # 4. AVALIAÇÃO DOS MODELOS
    logger.info("\n4. AVALIANDO MODELOS NO CONJUNTO DE TESTE")
    evaluator = CrossValidator(n_iterations=N_ITERATIONS)
    metrics_calculator = MetricsCalculator()

    all_results = {}
    all_accuracies = {}

    for model_name, model in models.items():
        logger.info(f"\nAvaliando {model_name}...")

        # Avalia modelo múltiplas vezes
        accuracies, all_metrics = evaluator.evaluate_model(
            model,
            best_params[model_name],
            X_train, y_train,
            X_test, y_test
        )

        # Armazena resultados
        all_accuracies[model_name] = accuracies
        all_results[model_name] = {
            'best_params': best_params[model_name],
            'accuracies': accuracies,
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'all_metrics': all_metrics,
            'final_metrics': all_metrics[-1]  # Métricas da última iteração
        }

        logger.info(f"{model_name} - Acurácia média: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

    # 5. ANÁLISE ESTATÍSTICA
    logger.info("\n5. REALIZANDO ANÁLISE ESTATÍSTICA")
    stat_tester = StatisticalTester()

    # Teste de Kruskal-Wallis
    kw_stat, kw_pvalue = stat_tester.kruskal_wallis_test(all_accuracies)

    # Testes de Mann-Whitney
    mw_results = stat_tester.mann_whitney_tests(all_accuracies)

    statistical_results = {
        'kruskal_wallis': {
            'statistic': kw_stat,
            'p_value': kw_pvalue
        },
        'mann_whitney': mw_results
    }

    # 6. IDENTIFICAÇÃO DO MELHOR MODELO
    mean_accuracies = {model: np.mean(acc) for model, acc in all_accuracies.items()}
    best_model = max(mean_accuracies, key=mean_accuracies.get)
    all_results[best_model]['is_best'] = True

    logger.info(f"\nMELHOR MODELO: {best_model}")
    logger.info(f"Acurácia: {mean_accuracies[best_model]:.4f}")

    # 7. GERAÇÃO DE VISUALIZAÇÕES
    logger.info("\n7. GERANDO VISUALIZAÇÕES")
    plotter = Plotter()

    # Gráfico de comparação de acurácia
    plotter.plot_accuracy_comparison(all_accuracies, RESULTS_DIR / "plots")

    # Heatmap dos testes estatísticos
    plotter.plot_statistical_heatmap(mw_results, RESULTS_DIR / "plots")

    # Gráfico radar de métricas
    final_metrics = {
        model: results['final_metrics']
        for model, results in all_results.items()
    }
    plotter.plot_metrics_radar(final_metrics, RESULTS_DIR / "plots")

    # 8. GERAÇÃO DO RELATÓRIO
    logger.info("\n8. GERANDO RELATÓRIO FINAL")
    report_generator = ReportGenerator(RESULTS_DIR)

    # Gera relatório HTML
    report_path = report_generator.generate_html_report(
        all_results,
        statistical_results,
        best_model
    )

    # Salva resumo em JSON
    summary = {
        'execution_info': {
            'date': datetime.now().isoformat(),
            'total_time': format_time(time.time() - start_time),
            'n_iterations': N_ITERATIONS,
            'data_split': {
                'train': TRAIN_SIZE,
                'validation': VAL_SIZE,
                'test': TEST_SIZE
            }
        },
        'results': all_results,
        'statistical_analysis': statistical_results,
        'best_model': best_model
    }

    summary_path = report_generator.save_summary_json(summary)

    # Finalização
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("EXECUÇÃO CONCLUÍDA COM SUCESSO!")
    logger.info(f"Tempo total: {format_time(total_time)}")
    logger.info(f"Relatório salvo em: {report_path}")
    logger.info(f"Resumo salvo em: {summary_path}")
    logger.info("=" * 80)

    print(f"\n✅ Execução concluída! Verifique os resultados em: {RESULTS_DIR}")


if __name__ == "__main__":
    main()