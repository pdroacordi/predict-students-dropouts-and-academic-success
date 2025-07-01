"""
Script principal
"""
import time

import numpy as np

from src.config.settings import *
from src.data.loader import DataLoader
from src.evaluation.cross_validation import CrossValidator
from src.evaluation.statistical_tests import StatisticalTester
from src.models.decision_tree_model import DecisionTreeModel
from src.models.knn_model import KNNModel
from src.models.mlp_model import MLPModel
from src.models.svm_model import SVMModel
from src.utils.helpers import save_results
from src.utils.logger import setup_logger
from src.visualization.plots import Plotter
from src.visualization.report_generator import ReportGenerator


def main():
    """Pipeline principal"""

    # Setup
    start_time = time.time()
    logger = setup_logger("ML_Comparison", log_dir=DATA_DIR, level=LOG_LEVEL)
    logger.info("=== INICIANDO PIPELINE ===")

    # Criar diretórios
    for directory in [PROCESSED_DATA_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento dos dados
    logger.info("1. Carregando dados...")
    data_loader = DataLoader(RAW_DATA_DIR / "students_dropout.csv")  # Ajuste o nome do arquivo
    df = data_loader.load_data()
    X_original, y_original = data_loader.get_features_and_target(df)

    logger.info(f"Dataset carregado: {X_original.shape} features, {len(np.unique(y_original))} classes")

    # 2. Inicializar modelos
    logger.info("2. Inicializando modelos...")
    models = {
        'knn': KNNModel(),
        'decision_tree': DecisionTreeModel(),
        'svm': SVMModel(),
        'mlp': MLPModel()
    }

    # 3. Avaliação com divisões independentes
    logger.info("3. Iniciando avaliação com divisões independentes...")
    cross_validator = CrossValidator(
        n_iterations=N_ITERATIONS,
        random_state=RANDOM_STATE,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE
    )

    results = {}
    all_best_params = {}

    for model_name, model in models.items():
        logger.info(f"\n--- Avaliando {model_name.upper()} ---")

        accuracies, metrics, best_params_list = cross_validator.evaluate_model(
            model, X_original, y_original
        )

        # Métricas finais (média das 10 execuções)
        final_metrics = {
            'accuracy': float(np.mean(accuracies)),
            'accuracy_std': float(np.std(accuracies)),
            'precision': float(np.mean([m['precision'] for m in metrics])),
            'recall': float(np.mean([m['recall'] for m in metrics])),
            'f1_score': float(np.mean([m['f1_score'] for m in metrics]))
        }

        results[model_name] = {
            'accuracies': accuracies,
            'final_metrics': final_metrics,
            'all_metrics': metrics
        }

        all_best_params[model_name] = best_params_list

        logger.info(f"{model_name} - Acurácia: {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}")

    # 4. Identificar melhor modelo
    logger.info("\n4. Identificando melhor modelo...")
    best_model = max(results.keys(), key=lambda k: results[k]['final_metrics']['accuracy'])
    logger.info(f"Melhor modelo: {best_model} (Acurácia: {results[best_model]['final_metrics']['accuracy']:.4f})")

    # 5. Análise estatística
    logger.info("\n5. Realizando análise estatística...")
    accuracies_dict = {model: results[model]['accuracies'] for model in results}

    statistical_tester = StatisticalTester(alpha=0.05)

    # Teste de Kruskal-Wallis
    kw_stat, kw_pvalue = statistical_tester.kruskal_wallis_test(accuracies_dict)

    # Testes de Mann-Whitney
    mw_results = statistical_tester.mann_whitney_tests(accuracies_dict)

    statistical_results = {
        'kruskal_wallis': {
            'statistic': float(kw_stat),
            'p_value': float(kw_pvalue)
        },
        'mann_whitney': mw_results
    }

    # 6. Visualizações
    logger.info("\n6. Gerando visualizações...")
    plotter = Plotter()

    # Boxplot de acurácias
    plotter.plot_accuracy_comparison(accuracies_dict, RESULTS_DIR)

    # Heatmap de testes estatísticos
    plotter.plot_statistical_heatmap(mw_results, RESULTS_DIR)

    # Radar de métricas
    metrics_for_radar = {
        model: results[model]['final_metrics']
        for model in results
    }
    plotter.plot_metrics_radar(metrics_for_radar, RESULTS_DIR)

    # 7. Relatório
    logger.info("\n7. Gerando relatório...")
    report_generator = ReportGenerator(RESULTS_DIR)

    # Marca o melhor modelo
    for model in results:
        results[model]['is_best'] = (model == best_model)
        results[model]['best_params'] = all_best_params[model][0]  # Primeiro conjunto como exemplo

    html_report = report_generator.generate_html_report(
        results, statistical_results, best_model
    )

    # 8. Salvar resultados
    logger.info("\n8. Salvando resultados...")

    final_results = {
        'experiment_info': {
            'n_iterations': N_ITERATIONS,
            'train_size': TRAIN_SIZE,
            'val_size': VAL_SIZE,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE
        },
        'models_performance': results,
        'best_model': best_model,
        'statistical_analysis': statistical_results,
        'best_parameters': all_best_params,
        'execution_time_seconds': time.time() - start_time
    }

    save_results(final_results, RESULTS_DIR / "final_results.json")
    report_generator.save_summary_json(final_results)

    # 9. Resumo final
    end_time = time.time()
    execution_time = end_time - start_time

    logger.info("\n" + "="*50)
    logger.info("RESUMO FINAL")
    logger.info("="*50)
    logger.info(f"Melhor modelo: {best_model}")
    logger.info(f"Acurácia: {results[best_model]['final_metrics']['accuracy']:.4f} ± {results[best_model]['final_metrics']['accuracy_std']:.4f}")
    logger.info(f"Teste Kruskal-Wallis p-valor: {kw_pvalue:.4f}")
    logger.info(f"Tempo de execução: {execution_time:.2f} segundos")
    logger.info(f"Relatório HTML: {html_report}")
    logger.info("="*50)

    return final_results


if __name__ == "__main__":
    results = main()