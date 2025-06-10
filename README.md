# Comparação de Algoritmos de Machine Learning

## 📋 Descrição

Este projeto implementa uma comparação completa e estatisticamente rigorosa entre quatro algoritmos de Machine Learning para classificação:

- **K-Nearest Neighbors (KNN)**
- **Árvores de Decisão** 
- **Support Vector Machine (SVM)**
- **Multi-Layer Perceptron (MLP)**

O projeto utiliza o dataset "Predict Students' Dropout and Academic Success" e realiza uma análise completa incluindo:
- Otimização de hiperparâmetros
- Avaliação com múltiplas execuções
- Testes estatísticos (Kruskal-Wallis e Mann-Whitney)
- Geração automática de relatórios e visualizações

## 🚀 Funcionalidades

- ✅ **Pipeline completo de ML**: desde o carregamento dos dados até a geração de relatórios
- ✅ **Tuning automático de hiperparâmetros** usando GridSearchCV
- ✅ **Avaliação robusta** com 10 execuções independentes
- ✅ **Análise estatística** para comparação significativa entre modelos
- ✅ **Visualizações profissionais**: boxplots, heatmaps e gráficos radar
- ✅ **Relatório HTML automático** com todos os resultados
- ✅ **Logging completo** de todas as etapas

## 📁 Estrutura do Projeto

```
ml-algorithms-comparison/
│
├── main.py                          # Script principal
├── requirements.txt                 # Dependências do projeto
├── README.md                        # Este arquivo
│ 
├── src/                             # Código fonte
│   ├── config/                      # Configurações
│   │   └── settings.py              # Parâmetros globais
│   │       
│   ├── data/                        # Processamento de dados
│   │   ├── loader.py                # Carregamento de dados
│   │   ├── preprocessor.py          # Pré-processamento
│   │   └── splitter.py              # Divisão treino/validação/teste
│   │       
│   ├── models/                      # Implementação dos modelos
│   │   ├── base_model.py            # Classe base
│   │   ├── knn_model.py             # K-Nearest Neighbors
│   │   ├── decision_tree_model.py   # Árvore de Decisão
│   │   ├── svm_model.py             # Support Vector Machine
│   │   └── mlp_model.py             # Multi-Layer Perceptron
│   │       
│   ├── tuning/                      # Otimização de hiperparâmetros
│   │   └── hyperparameter_tuning.py
│   │
│   ├── evaluation/                  # Avaliação e métricas
│   │   ├── cross_validation.py      # Validação cruzada
│   │   ├── metrics.py               # Cálculo de métricas
│   │   └── statistical_tests.py     # Testes estatísticos
│   │
│   ├── visualization/               # Visualizações e relatórios
│   │   ├── plots.py                 # Gráficos
│   │   └── report_generator.py      # Gerador de relatórios
│   │
│   └── utils/                       # Utilitários
│       ├── logger.py                # Sistema de logging
│       └── helpers.py               # Funções auxiliares
│       
└── data/                            # Diretório de dados
    ├── raw/                         # Dados originais
    ├── processed/                   # Dados processados
    └── results/                     # Resultados gerados
        ├── hyperparameters/         # Melhores parâmetros
        ├── metrics/                 # Métricas calculadas
        ├── plots/                   # Gráficos gerados
        └── *.html                   # Relatórios HTML
```

## 🔧 Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)
- Git (opcional, para clonar o repositório)

## 📦 Instalação

### 1. Clone o repositório (ou baixe o ZIP)

```bash
git clone https://github.com/seu-usuario/ml-algorithms-comparison.git
cd ml-algorithms-comparison
```

### 2. Crie um ambiente virtual (recomendado)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

## 📊 Dataset

### Estrutura esperada do dataset

O dataset deve conter:
- Features: características dos estudantes (demográficas, acadêmicas, socioeconômicas)
- Target: coluna "Target" indicando a situação do estudante (Dropout, Enrolled, Graduate)

## 🏃‍♂️ Como Executar

### 1. Certifique-se de que o ambiente virtual está ativado

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 2. Execute o pipeline completo

```bash
python main.py
```

### 3. Acompanhe o progresso

O script irá:
1. Carregar e pré-processar os dados
2. Dividir em 50% treino, 25% validação, 25% teste
3. Realizar tuning de hiperparâmetros para cada modelo
4. Avaliar cada modelo 10 vezes
5. Executar testes estatísticos
6. Gerar visualizações
7. Criar relatório HTML completo

### 4. Tempo estimado de execução

- **Total**: 1 minuto (dependendo do hardware)

## 📈 Hiperparâmetros Otimizados

### KNN
- **n_neighbors**: [3, 5, 7, 9, 11, 15]
- **weights**: ['uniform', 'distance']

### Árvore de Decisão
- **criterion**: ['gini', 'entropy']
- **max_depth**: [None, 5, 10, 15, 20]
- **min_samples_split**: [2, 5, 10]
- **min_samples_leaf**: [1, 2, 4]

### SVM
- **C**: [0.1, 1, 10, 100]
- **kernel**: ['poly', 'rbf']
- **degree**: [2, 3, 4] (apenas para kernel polinomial)
- **gamma**: ['scale', 'auto']

### MLP
- **hidden_layer_sizes**: [(50,), (100,), (50, 50), (100, 50)]
- **activation**: ['identity', 'logistic', 'tanh', 'relu']
- **learning_rate_init**: [0.001, 0.01, 0.1]
- **max_iter**: [200, 500, 1000]

## 📊 Resultados

Após a execução, você encontrará em `data/results/`:

### 1. Relatório HTML (`relatorio_completo_*.html`)
- Resumo executivo com o melhor modelo
- Tabela com melhores hiperparâmetros
- Métricas detalhadas (acurácia, precisão, recall, F1-score)
- Resultados dos testes estatísticos
- Conclusões e recomendações

### 2. Visualizações (`plots/`)
- **accuracy_comparison.png**: Boxplot comparando acurácias
- **statistical_tests_heatmap.png**: Heatmap dos p-valores
- **metrics_radar.png**: Gráfico radar com múltiplas métricas

### 3. Arquivo JSON (`summary_*.json`)
- Todos os resultados em formato estruturado
- Útil para análises posteriores

### 4. Log detalhado (`ml_comparison_*.log`)
- Registro completo da execução
- Útil para debugging

## 🧪 Testes Estatísticos

### Kruskal-Wallis
- Testa se há diferença significativa entre os grupos
- H0: Todos os modelos têm o mesmo desempenho
- Se p-valor < 0.05, rejeita H0

### Mann-Whitney
- Comparações par a par entre modelos
- Identifica quais modelos são significativamente diferentes
- Correção de Bonferroni aplicada para múltiplas comparações

## 🛠️ Personalização

### Modificar parâmetros globais

Edite `src/config/settings.py`:

```python
# Divisão dos dados
TRAIN_SIZE = 0.5  # 50% para treino
VAL_SIZE = 0.25   # 25% para validação
TEST_SIZE = 0.25  # 25% para teste

# Número de iterações para avaliação
N_ITERATIONS = 10

# Random state para reprodutibilidade
RANDOM_STATE = 42
```

### Adicionar novos modelos

1. Crie um arquivo em `src/models/novo_modelo.py`
2. Herde da classe `BaseModel`
3. Implemente `get_param_grid()` e `create_model()`
4. Adicione ao dicionário de modelos em `main.py`

## 🐛 Solução de Problemas

### Erro: "No module named 'src'"
- Execute o `main.py` da raiz do projeto, não de dentro de `src/`

### Erro: "File not found: students_dropout.csv"
- Verifique se o arquivo está em `data/raw/students_dropout.csv`
- Confirme se o nome do arquivo está correto

### Memória insuficiente
- Reduza o número de combinações de hiperparâmetros
- Use um subset dos dados para teste
- Execute um modelo por vez

### Tempo de execução muito longo
- Reduza `N_ITERATIONS` em settings.py
- Simplifique o grid de hiperparâmetros
- Use `n_jobs=1` se tiver problemas com paralelização

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE.txt) para mais detalhes.

---

**Nota**: Este projeto foi desenvolvido para fins educacionais e de pesquisa em Machine Learning.