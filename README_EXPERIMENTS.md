# Guia de Experimentação - YouTubeSafeKids

Este guia explica como usar os scripts de experimentação e treinamento dos modelos BERTimbau do YouTubeSafeKids.

## 📁 Estrutura dos Scripts

```
scripts/
├── train_model.py           # Treinamento individual de modelos
├── evaluate_model.py        # Avaliação de modelos treinados
├── data_preprocessing.py    # Pré-processamento de dados
├── model_comparison.py      # Comparação entre modelos
└── run_experiments.py       # Orquestração de experimentos completos

config/
└── experiment_config.json   # Configurações de experimento
```

## 🚀 Início Rápido

### 1. Teste Rápido

Para um teste rápido do sistema:

```bash
python scripts/run_experiments.py --quick-test --models sentiment
```

### 2. Experimento Completo

Para executar um experimento completo:

```bash
python scripts/run_experiments.py --config config/experiment_config.json
```

## 📊 Scripts Individuais

### Pré-processamento de Dados

```bash
# Pré-processar dados para modelo de sentimento
python scripts/data_preprocessing.py \
    --input data/sentiment_raw.csv \
    --output data/sentiment_processed.csv \
    --model sentiment \
    --clean-text \
    --balance \
    --split

# Parâmetros principais:
# --clean-text: Limpa e normaliza o texto
# --balance: Balanceia as classes
# --split: Divide em train/test
# --min-length: Comprimento mínimo do texto (padrão: 10)
# --max-length: Comprimento máximo do texto (padrão: 512)
```

### Treinamento de Modelos

```bash
# Treinar modelo de sentimento (configuração do artigo WICS 2026)
python scripts/train_model.py \
    --model sentiment \
    --train-data data/sentiment_train.csv \
    --output-dir models/sentiment \
    --epochs 5 \
    --batch-size 8 \
    --learning-rate 3e-5 \
    --warmup-steps 100

# Treinar modelo de toxicidade
python scripts/train_model.py \
    --model toxicity \
    --train-data data/toxicity_train.csv \
    --output-dir models/toxicity \
    --epochs 4 \
    --batch-size 16 \
    --learning-rate 3e-5 \
    --mixed-precision \
    --early-stopping
```

### Ensemble dos Folds (configuração do artigo)

O classificador final do artigo é um **ensemble (soft voting) dos cinco modelos** gerados pela validação cruzada. O script de avaliação do ensemble está em `_baselines/ensemble_folds.py`; os resultados consolidados em `_baselines/ensemble_results.json`.

```bash
python _baselines/ensemble_folds.py
```

### Avaliação de Modelos

```bash
# Avaliar modelo treinado
python scripts/evaluate_model.py \
    --model sentiment \
    --test-data data/sentiment_test.csv \
    --model-path models/sentiment/best_model.pt \
    --output-dir results/sentiment \
    --save-predictions

# Avaliar todos os tipos de modelo
for model in sentiment toxicity language educational; do
    python scripts/evaluate_model.py \
        --model $model \
        --test-data data/${model}_test.csv \
        --model-path models/${model}/best_model.pt \
        --output-dir results/${model}
done
```

### Comparação de Modelos

```bash
# Comparar diferentes versões de um modelo
python scripts/model_comparison.py \
    --models models/sentiment/v1.pt models/sentiment/v2.pt \
    --model-names "Versão 1" "Versão 2" \
    --test-data data/sentiment_test.csv \
    --model-type sentiment \
    --output comparison_sentiment \
    --save-predictions
```

## ⚙️ Configuração de Experimentos

O arquivo `config/experiment_config.json` contém todas as configurações:

### Configurações de Modelo

```json
{
  "models": {
    "sentiment": {
      "model_name": "neuralmind/bert-base-portuguese-cased",
      "num_labels": 3,
      "max_length": 512,
      "learning_rate": 3e-5,
      "batch_size": 8,
      "epochs": 5,
      "warmup_steps": 100,
      "dropout": 0.1
    }
  }
}
```

### Configurações de Dados

```json
{
  "data": {
    "preprocessing": {
      "clean_text": true,
      "remove_duplicates": true,
      "min_length": 10,
      "max_length": 512,
      "balance_classes": true
    },
    "split": {
      "test_size": 0.2,
      "validation_size": 0.1,
      "stratify": true
    }
  }
}
```

### Configurações de Treinamento

```json
{
  "training": {
    "mixed_precision": true,
    "early_stopping": {
      "enabled": true,
      "patience": 3,
      "monitor": "val_f1"
    },
    "scheduler": {
      "type": "linear",
      "warmup_ratio": 0.1
    }
  }
}
```

## 📈 Experimentos Automatizados

### Experimento Básico

```bash
# Executar experimento com configuração padrão
python scripts/run_experiments.py
```

### Experimento Personalizado

```bash
# Experimento com modelos específicos
python scripts/run_experiments.py \
    --models sentiment toxicity \
    --data-dir /path/to/data \
    --output-dir /path/to/results

# Experimento sem treinamento (apenas avaliação)
python scripts/run_experiments.py \
    --skip-training \
    --models sentiment

# Dry run (mostrar comandos sem executar)
python scripts/run_experiments.py \
    --dry-run \
    --verbose
```

### Teste Rápido

```bash
# Teste com configurações reduzidas
python scripts/run_experiments.py \
    --quick-test \
    --models sentiment
```

## 📋 Formato dos Dados

### Estrutura Esperada dos Arquivos CSV

```csv
text,label
"Este é um texto positivo",positive
"Este é um texto negativo",negative
"Este é um texto neutro",neutral
```

### Rótulos por Tipo de Modelo

- **Sentiment**: `Positivo`, `Negativo`, `Neutro`
- **Toxicity**: `toxic`, `non_toxic`
- **Language**: `appropriate`, `inappropriate`
- **Educational**: `educational`, `non_educational`

## 📊 Resultados e Relatórios

### Estrutura de Saída

```
results/
└── ExperimentName_20240101_120000/
    ├── models/                 # Modelos treinados
    │   ├── sentiment/
    │   ├── toxicity/
    │   └── ...
    ├── results/               # Resultados de avaliação
    │   ├── sentiment/
    │   │   ├── evaluation_results.json
    │   │   ├── predictions.csv
    │   │   └── confusion_matrix.png
    │   └── ...
    ├── reports/               # Relatórios de comparação
    │   ├── sentiment_comparison_report.json
    │   ├── sentiment_comparison_main_metrics.png
    │   └── ...
    ├── logs/                  # Logs de execução
    ├── data/                  # Dados processados
    ├── final_report.json      # Relatório final
    └── experiment_summary.txt # Resumo do experimento
```

### Métricas Disponíveis

- **Accuracy**: Precisão geral
- **Precision**: Precisão por classe
- **Recall**: Revocação por classe
- **F1-Score**: Média harmônica de precisão e revocação
- **AUC-ROC**: Área sob a curva ROC (modelos binários)
- **Confusion Matrix**: Matriz de confusão
- **Confidence Distribution**: Distribuição de confiança

## 🔧 Solução de Problemas

### Problemas Comuns

1. **Erro de memória GPU**:
   ```bash
   # Reduzir batch size
   python scripts/train_model.py --batch-size 8
   ```

2. **Dados não encontrados**:
   ```bash
   # Verificar estrutura de diretórios
   ls -la data/
   ```

3. **Modelo não carrega**:
   ```bash
   # Verificar caminho do modelo
   python -c "import torch; print(torch.load('models/sentiment/best_model.pt', map_location='cpu'))"
   ```

### Logs e Debugging

```bash
# Executar com logging verboso
python scripts/run_experiments.py --verbose

# Verificar logs
tail -f logs/experiment.log
```

## 🎯 Exemplos de Uso

### Exemplo 1: Treinamento Completo

```bash
# 1. Pré-processar dados
python scripts/data_preprocessing.py \
    --input data/sentiment_raw.csv \
    --output data/sentiment_processed.csv \
    --model sentiment \
    --clean-text --balance --split

# 2. Treinar modelo
python scripts/train_model.py \
    --model sentiment \
    --train-data data/sentiment_processed_train.csv \
    --output-dir models/sentiment_v1 \
    --epochs 5

# 3. Avaliar modelo
python scripts/evaluate_model.py \
    --model sentiment \
    --test-data data/sentiment_processed_test.csv \
    --model-path models/sentiment_v1/best_model.pt \
    --output-dir results/sentiment_v1
```

### Exemplo 2: Comparação de Modelos

```bash
# Treinar duas versões
python scripts/train_model.py \
    --model sentiment \
    --train-data data/sentiment_train.csv \
    --output-dir models/sentiment_lr2e5 \
    --learning-rate 2e-5

python scripts/train_model.py \
    --model sentiment \
    --train-data data/sentiment_train.csv \
    --output-dir models/sentiment_lr3e5 \
    --learning-rate 3e-5

# Comparar
python scripts/model_comparison.py \
    --models models/sentiment_lr2e5/best_model.pt models/sentiment_lr3e5/best_model.pt \
    --model-names "LR 2e-5" "LR 3e-5" \
    --test-data data/sentiment_test.csv \
    --model-type sentiment \
    --output comparison_lr
```

### Exemplo 3: Experimento Automatizado

```bash
# Criar configuração personalizada
cp config/experiment_config.json config/my_experiment.json
# Editar configurações...

# Executar experimento
python scripts/run_experiments.py \
    --config config/my_experiment.json \
    --models sentiment toxicity
```

## 📚 Recursos Adicionais

- **Configuração de GPU**: Ajustar `CUDA_VISIBLE_DEVICES`
- **Monitoramento**: Usar TensorBoard ou Weights & Biases
- **Otimização**: Técnicas de quantização e pruning
- **Deploy**: Scripts de conversão para produção

## 🤝 Contribuição

Para contribuir com melhorias nos scripts:

1. Teste suas modificações
2. Documente mudanças
3. Mantenha compatibilidade
4. Adicione testes unitários

---

Para mais informações, consulte a documentação principal do projeto ou abra uma issue no repositório.