"""
Grid search: LR = 3e-5 com variações de warmup e epochs.

Configs testadas:
  A: lr=3e-5, warmup_steps=100, epochs=5
  B: lr=3e-5, warmup_steps=500, epochs=5
  C: lr=3e-5, warmup_steps=500, epochs=8

Metodologia identica ao train_sentiment_melhorado.py:
  - 5-fold CV estratificado nos 80% de treino
  - SMOTE (Random Oversample) por fold
  - Holdout 20% avaliado apenas ao final, com o melhor fold de cada config

Saida:
  src/_baselines/grid_3e5_results.json  (nao sobrescreve nenhum resultado existente)
  Modelos em: src/app/nlp/models/trained/AS_grid_<X>_fold_<N>_<timestamp>/

Execucao:
  cd <projeto>/src
  python -m app.nlp.training.train_grid_3e5
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── injetar configs no dicionario TRAINING_CONFIG antes de qualquer outro import
# Python dicts sao mutaveis e o modulo e singleton, entao a alteracao e visivel
# em toda a aplicacao sem modificar config.py.
from app.nlp.config import TRAINING_CONFIG

_BASE_3E5 = {
    'per_device_train_batch_size': 8,
    'per_device_eval_batch_size': 8,
    'learning_rate': 3e-5,
    'weight_decay': 0.01,
    'logging_steps': 10,
    'eval_steps': 500,
    'evaluation_strategy': 'steps',
    'save_strategy': 'steps',   # sera sobrescrito por train_model() no Windows
    'save_steps': 500,
    'load_best_model_at_end': True,  # idem
    'metric_for_best_model': 'eval_f1',
    'greater_is_better': True,
    'seed': 42,
}

TRAINING_CONFIG['grid_A'] = {**_BASE_3E5, 'num_train_epochs': 5, 'warmup_steps': 100}
TRAINING_CONFIG['grid_B'] = {**_BASE_3E5, 'num_train_epochs': 5, 'warmup_steps': 500}
TRAINING_CONFIG['grid_C'] = {**_BASE_3E5, 'num_train_epochs': 8, 'warmup_steps': 500}

# ── demais imports apos injecao das configs ───────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.utils import resample

from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.utils.data_utils import get_kfold_split

# ── constantes ────────────────────────────────────────────────────────────────
K_FOLDS  = 5
LABELS   = ['Negativo', 'Neutro', 'Positivo']
OUT_FILE = Path(project_root) / '_baselines' / 'grid_3e5_results.json'

# Resultados do modelo de referencia (fold 1, smote, 5e-5/wr500/ep5)
REFERENCIA = {
    'name': 'Referencia: lr=5e-5, warmup=500, epochs=5',
    'holdout': {
        'accuracy':   0.7727,
        'f1_macro':   0.7749,
        'per_class': {
            'Negativo': {'f1': 0.9181},
            'Neutro':   {'f1': 0.7273},
            'Positivo': {'f1': 0.6792},
        }
    },
    'val_mean_f1': 0.7749,
    'val_std_f1':  0.0074,
}

GRID = [
    {'name': 'A', 'config_name': 'grid_A', 'lr': 3e-5, 'warmup': 100, 'epochs': 5},
    {'name': 'B', 'config_name': 'grid_B', 'lr': 3e-5, 'warmup': 500, 'epochs': 5},
    {'name': 'C', 'config_name': 'grid_C', 'lr': 3e-5, 'warmup': 500, 'epochs': 8},
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── balanceamento ─────────────────────────────────────────────────────────────
def apply_smote(train_texts: list, train_labels: list) -> tuple[list, list]:
    """Identico ao apply_smote de train_sentiment_melhorado.py."""
    df = pd.DataFrame({'text': train_texts, 'label': train_labels})
    label_counts = Counter(train_labels)
    max_count = max(label_counts.values())
    balanced = []
    for label in sorted(label_counts.keys()):
        df_class = df[df['label'] == label]
        if len(df_class) < max_count:
            df_up = resample(df_class, replace=True,
                             n_samples=max_count, random_state=42)
            balanced.append(df_up)
        else:
            balanced.append(df_class)
    df_bal = (pd.concat(balanced, ignore_index=True)
              .sample(frac=1, random_state=42)
              .reset_index(drop=True))
    return df_bal['text'].tolist(), df_bal['label'].tolist()


# ── avaliacao no holdout ──────────────────────────────────────────────────────
def eval_holdout(model_path: str,
                 X_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
    """
    Carrega o modelo salvo e avalia no holdout.
    Mesmo metodo usado em run_all_02.py para consistencia.
    """
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tok   = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.to(device).eval()

    preds: list[int] = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch = [str(x) for x in X_test[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=128, return_tensors='pt').to(device)
            preds.extend(
                model(**enc).logits.argmax(dim=-1).cpu().numpy().tolist())

    y_pred = np.array(preds)
    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            'precision': float(precision_score(
                y_test, y_pred, labels=[i], average='macro', zero_division=0)),
            'recall':    float(recall_score(
                y_test, y_pred, labels=[i], average='macro', zero_division=0)),
            'f1':        float(f1_score(
                y_test, y_pred, labels=[i], average='macro', zero_division=0)),
        }
    print(classification_report(
        y_test, y_pred, target_names=LABELS, digits=4, zero_division=0))

    return {
        'accuracy':            float(accuracy_score(y_test, y_pred)),
        'f1_macro':            float(f1_score(y_test, y_pred, average='macro')),
        'f1_weighted':         float(f1_score(y_test, y_pred, average='weighted')),
        'precision_weighted':  float(precision_score(
            y_test, y_pred, average='weighted', zero_division=0)),
        'recall_weighted':     float(recall_score(
            y_test, y_pred, average='weighted', zero_division=0)),
        'per_class':           per_class,
    }


# ── execucao de uma config ────────────────────────────────────────────────────
def run_config(cfg: dict,
               X_train_cv: np.ndarray, y_train_cv: np.ndarray,
               X_test: np.ndarray,     y_test: np.ndarray) -> dict:
    logger.info(f"\n{'=' * 70}")
    logger.info(f"CONFIG {cfg['name']}  |  lr={cfg['lr']}  "
                f"warmup={cfg['warmup']}  epochs={cfg['epochs']}")
    logger.info(f"{'=' * 70}")

    fold_results: list[dict] = []

    for fold, (train_idx, val_idx) in enumerate(
            get_kfold_split(X_train_cv, y_train_cv, n_splits=K_FOLDS)):
        curr_fold = fold + 1
        logger.info(f"\n[Config {cfg['name']}] Fold {curr_fold}/{K_FOLDS}")

        X_ft = X_train_cv[train_idx]
        X_fv = X_train_cv[val_idx]
        y_ft = y_train_cv[train_idx]
        y_fv = y_train_cv[val_idx]

        logger.info(f"  Treino bruto: {len(X_ft)} | Validacao: {len(X_fv)}")
        X_bal, y_bal = apply_smote(X_ft.tolist(), y_ft.tolist())
        logger.info(f"  Treino apos SMOTE: {len(X_bal)}")

        model = BertimbauSentiment()
        results = model.train_model(
            train_texts=X_bal,
            train_labels=y_bal,
            val_texts=X_fv.tolist(),
            val_labels=y_fv.tolist(),
            config_name=cfg['config_name'],
            experiment_name=f'grid_{cfg["name"]}_fold_{curr_fold}',
        )

        metrics  = results['final_metrics']
        val_f1   = float(metrics.get('eval_f1', 0.0))
        val_acc  = float(metrics.get('eval_accuracy', 0.0))
        logger.info(f"  Fold {curr_fold} -> val_acc={val_acc:.4f}  val_f1={val_f1:.4f}")

        fold_results.append({
            'fold':         curr_fold,
            'model_path':   results['model_path'],
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1,
            'val_metrics':  {k: float(v) for k, v in metrics.items()
                             if isinstance(v, (int, float))},
        })

    # melhor fold por F1 macro na validacao
    best = max(fold_results, key=lambda x: x['val_f1_macro'])
    logger.info(f"\nMelhor fold: {best['fold']}  (val_f1={best['val_f1_macro']:.4f})")
    logger.info(f"Avaliando no holdout com modelo: {best['model_path']}")

    holdout = eval_holdout(best['model_path'], X_test, y_test)
    logger.info(f"Holdout -> acc={holdout['accuracy']:.4f}  "
                f"f1_macro={holdout['f1_macro']:.4f}")

    val_f1s = [f['val_f1_macro'] for f in fold_results]
    return {
        'config':          cfg,
        'folds':           fold_results,
        'best_fold':       best['fold'],
        'best_model_path': best['model_path'],
        'val_mean_f1':     float(np.mean(val_f1s)),
        'val_std_f1':      float(np.std(val_f1s)),
        'holdout':         holdout,
    }


# ── resumo comparativo ────────────────────────────────────────────────────────
def print_summary(all_results: list[dict]) -> None:
    print("\n" + "=" * 90)
    print("RESUMO COMPARATIVO")
    print("=" * 90)
    header = (f"{'Config':<38} {'Acc Holdout':>11} {'F1 Macro':>9}"
              f" {'F1 Neg':>7} {'F1 Neu':>7} {'F1 Pos':>7} {'Val F1 med+-std':>16}")
    print(header)
    print("-" * 90)

    # linha de referencia
    rh = REFERENCIA['holdout']
    pc = rh['per_class']
    print(f"{'Referencia (5e-5 / wr500 / ep5)':<38}"
          f" {rh['accuracy']*100:>10.2f}%"
          f" {rh['f1_macro']*100:>8.2f}%"
          f" {pc['Negativo']['f1']*100:>6.1f}%"
          f" {pc['Neutro']['f1']*100:>6.1f}%"
          f" {pc['Positivo']['f1']*100:>6.1f}%"
          f"  (modelo salvo)")

    for r in all_results:
        c  = r['config']
        h  = r['holdout']
        pc = h['per_class']
        label = (f"Config {c['name']}"
                 f" (lr=3e-5 / wr={c['warmup']} / ep={c['epochs']})")
        val_str = f"{r['val_mean_f1']*100:.2f}% +- {r['val_std_f1']*100:.2f}%"
        print(f"{label:<38}"
              f" {h['accuracy']*100:>10.2f}%"
              f" {h['f1_macro']*100:>8.2f}%"
              f" {pc['Negativo']['f1']*100:>6.1f}%"
              f" {pc['Neutro']['f1']*100:>6.1f}%"
              f" {pc['Positivo']['f1']*100:>6.1f}%"
              f"  {val_str:>16}")
    print("=" * 90)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()

    logger.info("Carregando corpus (split 80/20, seed=42)...")
    (X_cv_list, y_cv_list), (X_test_list, y_test_list) = get_data_for_cv_and_test()
    X_train_cv = np.array(X_cv_list,   dtype=object)
    y_train_cv = np.array(y_cv_list,   dtype=int)
    X_test     = np.array(X_test_list, dtype=object)
    y_test     = np.array(y_test_list, dtype=int)

    logger.info(f"Treino CV : {len(X_train_cv)} amostras")
    logger.info(f"Holdout   : {len(X_test)} amostras  "
                f"{dict(sorted(Counter(y_test).items()))}")

    all_results: list[dict] = []

    for cfg in GRID:
        result = run_config(cfg, X_train_cv, y_train_cv, X_test, y_test)
        all_results.append(result)
        _save_partial(all_results)    # salva apos cada config por seguranca

    print_summary(all_results)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Tempo total: {elapsed:.1f} min")
    logger.info(f"Resultados finais em: {OUT_FILE}")


def _save_partial(data: list[dict]) -> None:
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Resultados parciais salvos -> {OUT_FILE.name}")


if __name__ == '__main__':
    main()
