# Notas do Projeto — Análise de Sentimentos no YouTube para Conteúdo Infantil

**Artigo**: WICS 2026  
**Modelo base**: BERTimbau (`neuralmind/bert-base-portuguese-cased`)  
**Data de encerramento**: 16 de Maio de 2026

---

## 1. Corpus

### Construção e evolução

| Fase | Frases | Neg | Neu | Pos |
|------|--------|-----|-----|-----|
| Inicial | 2.610 | 39,1% | 46,4% | 11,7% |
| Final (artigo) | 2.749 | 31,6% | 36,3% | 32,1% |

Todas as frases do corpus são **reais**, extraídas de transcrições e títulos de vídeos do YouTube e rotuladas manualmente. Nenhuma frase foi gerada artificialmente.

**Ações realizadas para chegar ao corpus final:**
- Adição de 89 frases positivas reais (lote 1) e 180 frases positivas reais (lote 2) — todas coletadas do YouTube
- Revisão manual interativa de 395 frases com indicadores positivos fracos → 36 reclassificadas para Neutro, 11 para Negativo
- Remoção de duplicatas

### Divisão usada no artigo

- **Treino (CV)**: 2.199 frases (80%)
- **Teste holdout**: 550 frases (20%) — isoladas desde o início, nunca usadas no treino

---

## 2. Metodologia de Treinamento

### Configuração final (artigo)

- **Validação**: 5-Fold Stratified Cross-Validation sobre os 80% de treino
- **Balanceamento**: Random Oversampling aplicado **apenas nos folds de treino** (o holdout permanece intocado). O oversampling opera sobre as representações vetoriais usadas durante o treinamento — não modifica nem gera texto no corpus.
- **Otimizador**: AdamW com learning rate scheduling
- **Épocas**: 5
- **Batch size**: 8
- **Learning rate**: 3e-5
- **Warmup steps**: 100
- **Seed**: 42

### Nota sobre versões intermediárias

Os arquivos `RESULTADOS_FINAIS_SMOTE.md` e `RELATORIO_FINAL_COMPLETO.md` (agora removidos) reportavam acurácias de 84–85%. Esses números são de uma fase intermediária com o dataset ainda desbalanceado (Positivo = 11,7%). Com o dataset balanceado final, a tarefa ficou genuinamente mais difícil para todas as classes e os números são os reportados abaixo.

---

## 3. Resultados Finais (reportados no artigo)

### 5-Fold Cross-Validation (sobre os 80% de treino)

| Fold | F1 Macro |
|------|----------|
| 1 | 78,24% |
| 2 | 75,67% |
| 3 | 76,57% |
| 4 | 79,54% |
| 5 | 77,24% |
| **Média ± Std** | **77,46% ± 1,41%** |

Fonte: `_baselines/ensemble_results.json` (Config A: lr=3e-5, warmup=100, épocas=5).

### Avaliação no conjunto holdout (550 amostras, 20%)

O classificador final é o **ensemble (soft voting) dos cinco modelos** da validação cruzada.

| Métrica | Valor |
|---------|-------|
| Acurácia | **79,64%** |
| F1 Macro | **79,84%** |
| F1 Weighted | 79,63% |
| Precisão Macro | 81,46% |
| Recall Macro | 79,45% |

**Por classe (holdout):**

| Classe | Precisão | Recall | F1 | Suporte |
|--------|----------|--------|----|---------|
| Negativo | 93,4% | 89,1% | **91,2%** | 174 |
| Neutro | 69,0% | 84,5% | 76,0% | 200 |
| Positivo | 82,0% | 64,8% | 72,4% | 176 |

Fonte: `_baselines/ensemble_results.json` (Config A, campo `ensemble`).

---

## 4. Baselines Comparados

| Modelo | Acurácia | F1 Macro | F1 Neg | F1 Neu | F1 Pos |
|--------|----------|----------|--------|--------|--------|
| SentiLex-PT | 45,45% | 44,68% | 47,1% | 48,2% | 38,7% |
| TF-IDF + Regressão Logística | 69,45% | 69,66% | 80,2% | 65,3% | 63,4% |
| BERTimbau congelado + LR | 74,73% | 74,81% | 88,0% | 69,6% | 66,9% |
| **BERTimbau ensemble (proposto)** | **79,64%** | **79,84%** | **91,2%** | **76,0%** | **72,4%** |

Fonte: `_baselines/baselines_results.json` (três primeiros) e `_baselines/ensemble_results.json` (ensemble).

**Ganho sobre o baseline mais forte (BERTimbau congelado, 74,73%):**
- Acurácia: +4,91 pp
- F1 Macro: +5,03 pp
- F1 Negativo (crítico para segurança infantil): **91,2%** (+3,2 pp)

---

## 5. Experimentos Realizados (não reportados no artigo)

Durante o desenvolvimento foram conduzidos experimentos adicionais cujos modelos foram posteriormente removidos:

| Experimento | CV Accuracy | Observação |
|-------------|-------------|------------|
| CV básico sem SMOTE (5 folds) | ~82% | Dataset ainda desbalanceado |
| Variant A: weights sem SMOTE | 79,08% ± 1,59% | Melhor que SMOTE em CV, mas não testado em holdout de forma definitiva |
| Ablation sem negações | — | Estudo de ablação |
| Label Smoothing (fold 1) | 80,68% | Experimento pontual, não completou 5 folds |

---

## 6. Estrutura de Arquivos do Projeto

```
artigo_03/
├── src/
│   ├── app/
│   │   ├── nlp/
│   │   │   ├── models/
│   │   │   │   ├── trained/          ← modelos dos 5 folds (publicados no Hugging Face Hub)
│   │   │   │   └── cache/            ← BERTimbau base baixado do HuggingFace (~418 MB)
│   │   │   ├── datasets/             ← corpus.csv (dataset final)
│   │   │   ├── training/             ← scripts de treinamento
│   │   │   ├── evaluation/
│   │   │   │   └── results/          ← resultados de avaliação
│   │   │   └── utils/
│   │   ├── api/                      ← endpoints REST
│   │   ├── core/                     ← YouTube API, logging
│   │   ├── filters/                  ← filtros de sentimento, toxicidade, etc.
│   │   ├── static/ e templates/      ← protótipo web
│   │   └── main.py
│   ├── scripts/                      ← scripts organizados de treino e avaliação
│   ├── _baselines/
│   │   ├── baselines.py              ← avaliação dos baselines clássicos
│   │   ├── baselines_results.json    ← resultados dos 3 baselines citados no artigo
│   │   ├── ensemble_folds.py         ← avaliação do ensemble dos 5 folds no holdout
│   │   ├── ensemble_results.json     ← resultado do ensemble citado no artigo
│   │   ├── gen_confusion_matrix.py   ← geração da matriz de confusão (figura do artigo)
│   │   ├── run_all_02.py             ← orquestração dos baselines
│   │   ├── run_lexicon_baselines.py  ← baseline SentiLex-PT
│   │   ├── run_congelado_baseline.py ← baseline BERTimbau congelado
│   │   ├── reeval_grid_holdout.py    ← reavaliação de configurações de grid
│   │   ├── verify_article_corpus.py  ← verificação do corpus
│   │   └── SentiLex-flex-PT02.txt    ← léxico SentiLex-PT
│   ├── config/                       ← configuração de hiperparâmetros
│   └── data/                         ← corpus.csv (cópia para uso pelos scripts)
└── latex/                            ← artigo LaTeX (sbc-template.tex)
```

### Modelo no Hugging Face Hub

O modelo fine-tuned está publicado em:

**https://huggingface.co/ravarmes/bertimbau-sentiment-youtube-pt**

Os diretórios `models/trained/` e `models/cache/` estão excluídos do versionamento Git via `.gitignore` (arquivos `.safetensors` de ~420 MB cada).
