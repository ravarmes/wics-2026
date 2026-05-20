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

### Configuração final

- **Validação**: 5-Fold Stratified Cross-Validation sobre os 80% de treino
- **Balanceamento**: Random Oversampling aplicado **apenas nos folds de treino** (o holdout permanece intocado). O oversampling opera sobre as representações vetoriais usadas durante o treinamento — não modifica nem gera texto no corpus.
- **Otimizador**: AdamW com learning rate scheduling
- **Épocas**: 5
- **Batch size**: 8
- **Seed**: 42

### Nota sobre versões intermediárias

Os arquivos `RESULTADOS_FINAIS_SMOTE.md` e `RELATORIO_FINAL_COMPLETO.md` (agora removidos) reportavam acurácias de 84–85%. Esses números são de uma fase intermediária com o dataset ainda desbalanceado (Positivo = 11,7%). Com o dataset balanceado final, a tarefa ficou genuinamente mais difícil para todas as classes e os números são os reportados abaixo.

---

## 3. Resultados Finais (reportados no artigo)

### 5-Fold Cross-Validation (sobre os 80% de treino)

| Fold | Acurácia | F1 Macro | Precisão | Recall | Modelo salvo |
|------|----------|----------|----------|--------|--------------|
| 1 | 78,64% | 0,7886 | 0,7946 | 0,7864 | `AS_sentiment_cv_fold_1_smote_20260515_193838` |
| 2 | 76,59% | 0,7672 | 0,7732 | 0,7659 | `AS_sentiment_cv_fold_2_smote_20260515_215621` |
| 3 | 77,73% | 0,7787 | 0,7945 | 0,7773 | `AS_sentiment_cv_fold_3_smote_20260515_210251` |
| 4 | 77,73% | 0,7769 | 0,7912 | 0,7773 | `AS_sentiment_cv_fold_4_smote_20260515_230329` |
| 5 | 76,77% | 0,7690 | 0,7783 | 0,7677 | `AS_sentiment_cv_fold_5_smote_20260515_223343` |
| **Média ± Std** | **77,49% ± 0,74%** | **0,7761 ± 0,0076** | — | — | — |

### Avaliação no conjunto holdout (20% isolados)

O teste oficial foi realizado com o **modelo do Fold 1** (`193838`), escolhido por ser o de maior acurácia na validação.

> **Nota metodológica**: foi avaliado apenas o Fold 1 no holdout, não um ensemble dos 5 folds. Uma abordagem de ensemble (média das probabilidades dos 5 modelos) seria metodologicamente mais robusta, mas o resultado abaixo já está consolidado no artigo.

| Métrica | Valor |
|---------|-------|
| Acurácia | **77,27%** |
| F1 Macro | 0,7749 |
| F1 Weighted | 0,7723 |
| Precisão Macro | 0,7873 |
| Recall Macro | 0,7720 |

**Por classe (holdout):**

| Classe | Precisão | Recall | F1 | Suporte |
|--------|----------|--------|----|---------|
| Negativo | 0,9345 | 0,9023 | **0,9181** | 174 |
| Neutro | 0,6667 | 0,8000 | 0,7273 | 200 |
| Positivo | 0,7606 | 0,6136 | 0,6792 | 176 |

---

## 4. Baselines Comparados

| Modelo | Acurácia | F1 Macro |
|--------|----------|----------|
| Léxico simples (PT-BR) | 38,18% | 0,2978 |
| TF-IDF + Regressão Logística | 69,45% | 0,6966 |
| BERTimbau zero-shot + LogReg | 74,73% | 0,7481 |
| **BERTimbau fine-tuned (CV média)** | **77,49%** | **0,7761** |
| **BERTimbau fine-tuned (holdout)** | **77,27%** | **0,7749** |

**Ganho sobre o baseline mais forte (zero-shot, 74,73%):**
- Holdout: +2,54 pp em acurácia
- F1 Macro holdout: +2,68 pp (0,7749 vs 0,7481)
- F1 Negativo (crítico para segurança infantil): **0,9181 (91,8%)**

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
│   │   │   │   ├── trained/          ← modelo Fold 1 (publicado no Hugging Face Hub)
│   │   │   │   └── cache/            ← BERTimbau base baixado do HuggingFace (~418 MB)
│   │   │   ├── datasets/             ← corpus.csv (dataset final)
│   │   │   ├── training/             ← scripts de treinamento
│   │   │   ├── evaluation/
│   │   │   │   └── results/
│   │   │   │       ├── holdout_oficial.json      ← resultado do modelo no holdout (77,27%)
│   │   │   │       ├── baselines_por_classe.json ← baselines com métricas por classe
│   │   │   │       └── comparacao_final.json     ← comparação geral com CV folds
│   │   │   └── utils/
│   │   ├── api/                      ← endpoints REST
│   │   ├── core/                     ← YouTube API, logging
│   │   ├── filters/                  ← filtros de sentimento, toxicidade, etc.
│   │   ├── static/ e templates/      ← protótipo web
│   │   └── main.py
│   ├── scripts/                      ← scripts organizados de treino e avaliação
│   ├── _baselines/
│   │   ├── baselines.py              ← script de avaliação dos baselines
│   │   ├── baselines_results.json    ← resumo dos baselines (sem per-class completo)
│   │   ├── baseline_lexico.json      ← resultado individual do léxico
│   │   ├── baseline_tfidf.json       ← resultado individual do TF-IDF
│   │   └── baseline_bertimbau.json   ← resultado individual do BERTimbau zero-shot
│   ├── config/                       ← configuração de hiperparâmetros
│   └── data/                         ← corpus.csv (cópia para uso pelos scripts)
└── latex/                            ← artigo LaTeX (sbc-template.tex)
```

### Modelo no Hugging Face Hub

O modelo fine-tuned (Fold 1) está publicado em:

**https://huggingface.co/ravarmes/bertimbau-sentiment-youtube-pt**

Os diretórios `models/trained/` e `models/cache/` estão excluídos do versionamento Git via `.gitignore` (arquivos `.safetensors` de ~420 MB cada).
