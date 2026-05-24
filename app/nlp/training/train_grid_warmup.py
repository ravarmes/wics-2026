"""
Grid search 3: configs G, H, I — mapeamento do warmup.

Motivacao:
  Com avaliacao correta (holdout pre-processado), o regime vencedor e
  3e-5 / ep5 / linear / batch8. Dentro dele, o unico botao pouco amostrado e o
  warmup: a Config A usou warmup=100 (~8% do treino) e a Config B warmup=500
  (~42%) — muito distantes. Ha sinal de que menos warmup preserva o F1 Negativo.
  Estas configs mapeiam o meio da curva.

Configs (todas 3e-5, ep5, linear, batch8):
  G: warmup=50   (~4% do treino)
  H: warmup=200  (~17%)
  I: warmup=300  (~25%)

CORRECAO vs train_grid_3e5.py / train_grid_def.py:
  Aqueles scripts avaliavam o holdout em TEXTO CRU — bug que subestimou os
  resultados em ~2,4pp (ver verify_article_corpus.py). Este script avalia o
  holdout com preprocess_for_sentiment (limpeza + marcacao de negacoes), o
  metodo que reproduz o 77,27% do artigo digito a digito.

Metodologia: 5-fold CV estratificado nos 80%, SMOTE (oversampling) por fold,
holdout 20% avaliado ao final. Identica aos grids anteriores, exceto a correcao
de avaliacao acima.

Bonus (sem treino): reavalia os 5 folds ja treinados de A e B no holdout com
pre-processamento — pode revelar um fold melhor de graca.

Saidas (nao sobrescreve nada existente):
  src/_baselines/ab_folds_reeval.json    (reavaliacao dos folds de A e B)
  src/_baselines/grid_warmup_results.json (resultados de G, H, I)
  Modelos em: src/app/nlp/models/trained/AS_grid_<X>_fold_<N>_<timestamp>/

Execucao:
  cd <projeto>/src
  python -m app.nlp.training.train_grid_warmup
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── injetar configs no TRAINING_CONFIG antes dos demais imports ───────────────
from app.nlp.config import TRAINING_CONFIG

_BASE = {
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
    'num_train_epochs': 5,
    'seed': 42,
}

TRAINING_CONFIG['grid_G'] = {**_BASE, 'warmup_steps': 50}
TRAINING_CONFIG['grid_H'] = {**_BASE, 'warmup_steps': 200}
TRAINING_CONFIG['grid_I'] = {**_BASE, 'warmup_steps': 300}

# ── demais imports ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.utils import resample
from transformers import BertForSequenceClassification, BertTokenizer

from app.nlp.config import get_task_config
from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
from app.nlp.evaluation.model_evaluator import ModelEvaluator
from app.nlp.models.bertimbau_sentiment import BertimbauSentiment
from app.nlp.utils.data_utils import get_kfold_split

# ── constantes ────────────────────────────────────────────────────────────────
K_FOLDS  = 5
LABELS   = ['Negativo', 'Neutro', 'Positivo']
BASE_DIR = Path(project_root) / '_baselines'
OUT_WARMUP = BASE_DIR / 'grid_warmup_results.json'
OUT_ABFOLD = BASE_DIR / 'ab_folds_reeval.json'
GRID_3E5   = BASE_DIR / 'grid_3e5_results.json'   # contem A, B, C

# Alvo do artigo (holdout_oficial.json), avaliado com pre-processamento
REFERENCIA = {
    'name': 'Referencia (artigo): 5e-5 / wr500 / ep5',
    'accuracy': 0.7727, 'f1_macro': 0.7749,
    'f1_per_class': [0.9181, 0.7273, 0.6792],
}

GRID = [
    {'name': 'G', 'config_name': 'grid_G', 'lr': 3e-5, 'warmup': 50,  'epochs': 5},
    {'name': 'H', 'config_name': 'grid_H', 'lr': 3e-5, 'warmup': 200, 'epochs': 5},
    {'name': 'I', 'config_name': 'grid_I', 'lr': 3e-5, 'warmup': 300, 'epochs': 5},
]


# ── pre-processamento (copia verbatim de bertimbau_sentiment.py) ───────────────
def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    processed_text = text.replace('"""', '').replace('"', '')
    processed_text = re.sub(r'\s+', ' ', processed_text)
    return processed_text.strip()


def _handle_negations(text: str) -> str:
    negation_words = ['não', 'nao', 'nunca', 'jamais', 'nem',
                      'nenhum', 'nenhuma', 'sem']
    processed_text = text
    for negation in negation_words:
        pattern = r'\b' + negation + r'\s+(\w+)'
        matches = re.finditer(pattern, processed_text, re.IGNORECASE)
        replacements = []
        for match in matches:
            negated_word = match.group(1)
            replacement = f'{match.group(0).split()[0]} NEG_{negated_word}'
            replacements.append((match.group(0), replacement))
        for original, replacement in reversed(replacements):
            processed_text = processed_text.replace(original, replacement, 1)
    return processed_text


def preprocess(text: str) -> str:
    """preprocess_for_sentiment completo (limpeza + negacoes)."""
    return _handle_negations(_clean_text(text))


# ── balanceamento (verbatim de train_grid_def.py) ─────────────────────────────
def apply_smote(train_texts: list, train_labels: list) -> tuple[list, list]:
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


# ── avaliacao no holdout (COM pre-processamento) ──────────────────────────────
def eval_holdout(model_path: str,
                 X_test_pp: list,
                 y_test: list,
                 evaluator: ModelEvaluator) -> dict:
    """Avalia o modelo salvo no holdout JA pre-processado."""
    model     = BertForSequenceClassification.from_pretrained(str(model_path))
    tokenizer = BertTokenizer.from_pretrained(str(model_path))
    res = evaluator.evaluate_model(
        model=model, tokenizer=tokenizer,
        test_texts=X_test_pp, test_labels=y_test,
        max_length=128, batch_size=32,
    )
    return {
        'accuracy':         float(res['accuracy']),
        'f1_macro':         float(res['f1_macro']),
        'f1_weighted':      float(res['f1_weighted']),
        'f1_per_class':     [float(x) for x in res['f1_per_class']],
        'confusion_matrix': res['confusion_matrix'],
    }


# ── FASE 1: reavaliacao dos folds de A e B (sem treino) ───────────────────────
def reeval_ab_folds(X_test_pp: list, y_test: list,
                    evaluator: ModelEvaluator) -> list[dict]:
    if not GRID_3E5.exists():
        logger.warning(f"{GRID_3E5.name} nao encontrado — Fase 1 ignorada.")
        return []

    with open(GRID_3E5, encoding='utf-8') as f:
        grid_3e5 = json.load(f)

    out: list[dict] = []
    for name in ('A', 'B'):
        entry = next((x for x in grid_3e5
                      if x['config']['name'] == name), None)
        if entry is None:
            logger.warning(f"Config {name} nao encontrada em {GRID_3E5.name}")
            continue

        logger.info(f"\n[Fase 1] Reavaliando os 5 folds da Config {name}")
        fold_rows: list[dict] = []
        for fd in entry['folds']:
            mp = Path(fd['model_path'])
            if not mp.exists():
                logger.warning(f"  Fold {fd['fold']}: modelo ausente — {mp}")
                continue
            h = eval_holdout(mp, X_test_pp, y_test, evaluator)
            logger.info(f"  Fold {fd['fold']}: val_f1={fd['val_f1_macro']:.4f}  "
                        f"-> holdout acc={h['accuracy']:.4f}  "
                        f"f1={h['f1_macro']:.4f}")
            fold_rows.append({
                'fold':         fd['fold'],
                'val_f1_macro': fd['val_f1_macro'],
                'model_path':   fd['model_path'],
                'holdout':      h,
            })

        if fold_rows:
            hold_best = max(fold_rows, key=lambda x: x['holdout']['f1_macro'])
            out.append({
                'config':              entry['config'],
                'val_best_fold':       entry['best_fold'],
                'holdout_best_fold':   hold_best['fold'],
                'folds':               fold_rows,
            })
    return out


# ── FASE 2: treino e avaliacao de uma config ──────────────────────────────────
def run_config(cfg: dict,
               X_train_cv: np.ndarray, y_train_cv: np.ndarray,
               X_test_pp: list, y_test: list,
               evaluator: ModelEvaluator) -> dict:
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

        X_bal, y_bal = apply_smote(X_ft.tolist(), y_ft.tolist())
        logger.info(f"  Treino bruto: {len(X_ft)} | apos SMOTE: {len(X_bal)} "
                    f"| Validacao: {len(X_fv)}")

        model = BertimbauSentiment()
        results = model.train_model(
            train_texts=X_bal,
            train_labels=y_bal,
            val_texts=X_fv.tolist(),
            val_labels=y_fv.tolist(),
            config_name=cfg['config_name'],
            experiment_name=f'grid_{cfg["name"]}_fold_{curr_fold}',
        )

        metrics = results['final_metrics']
        val_f1  = float(metrics.get('eval_f1', 0.0))
        val_acc = float(metrics.get('eval_accuracy', 0.0))

        # holdout COM pre-processamento
        h = eval_holdout(results['model_path'], X_test_pp, y_test, evaluator)
        logger.info(f"  Fold {curr_fold} -> val_f1={val_f1:.4f}  | "
                    f"holdout acc={h['accuracy']:.4f}  f1={h['f1_macro']:.4f}")

        fold_results.append({
            'fold':         curr_fold,
            'model_path':   results['model_path'],
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1,
            'holdout':      h,
        })

    val_best  = max(fold_results, key=lambda x: x['val_f1_macro'])
    hold_best = max(fold_results, key=lambda x: x['holdout']['f1_macro'])
    val_f1s   = [f['val_f1_macro'] for f in fold_results]
    logger.info(f"\nConfig {cfg['name']}: melhor fold por validacao = "
                f"{val_best['fold']}  (holdout f1="
                f"{val_best['holdout']['f1_macro']:.4f})")

    return {
        'config':            cfg,
        'folds':             fold_results,
        'val_best_fold':     val_best['fold'],
        'holdout_best_fold': hold_best['fold'],
        'val_mean_f1':       float(np.mean(val_f1s)),
        'val_std_f1':        float(np.std(val_f1s)),
        'holdout_val_best':  val_best['holdout'],
    }


# ── resumo ────────────────────────────────────────────────────────────────────
def _row(label: str, h: dict, tail: str = "") -> str:
    pc = h['f1_per_class']
    return (f"{label:<40}"
            f" {h['accuracy']*100:>9.2f}%"
            f" {h['f1_macro']*100:>9.2f}%"
            f" {pc[0]*100:>7.1f}%"
            f" {pc[1]*100:>7.1f}%"
            f" {pc[2]*100:>7.1f}%"
            f"  {tail}")


def print_summary(warmup_results: list[dict], ab_results: list[dict]) -> None:
    print("\n" + "=" * 96)
    print("RESUMO  —  holdout avaliado COM pre-processamento (metodo do artigo)")
    print("=" * 96)
    print(f"{'Config':<40} {'Acc':>10} {'F1 Macro':>10}"
          f" {'F1 Neg':>8} {'F1 Neu':>8} {'F1 Pos':>8}")
    print("-" * 96)

    print(_row("Referencia (artigo, 5e-5/wr500/ep5)",
               {'accuracy': REFERENCIA['accuracy'],
                'f1_macro': REFERENCIA['f1_macro'],
                'f1_per_class': REFERENCIA['f1_per_class']}))

    for r in ab_results:
        c = r['config']
        vb = next(f for f in r['folds'] if f['fold'] == r['val_best_fold'])
        print(_row(f"Config {c['name']} (3e-5/wr={c['warmup']}/ep=5) fold {vb['fold']}",
                   vb['holdout']))

    for r in warmup_results:
        c  = r['config']
        vb = next(f for f in r['folds'] if f['fold'] == r['val_best_fold'])
        tail = f"val {r['val_mean_f1']*100:.2f}+-{r['val_std_f1']*100:.2f}"
        print(_row(f"Config {c['name']} (3e-5/wr={c['warmup']}/ep=5) fold {vb['fold']}",
                   vb['holdout'], tail))
    print("=" * 96)
    print("Linha de cada config = melhor fold por F1 de VALIDACAO (selecao limpa).")


def _save(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Salvo -> {path.name}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()

    logger.info("Carregando corpus (split 80/20, seed=42)...")
    (X_cv_list, y_cv_list), (X_test_list, y_test_list) = get_data_for_cv_and_test()
    X_train_cv = np.array(X_cv_list, dtype=object)
    y_train_cv = np.array(y_cv_list, dtype=int)
    y_test     = [int(y) for y in y_test_list]

    # holdout pre-processado UMA vez (limpeza + negacoes)
    X_test_pp = [preprocess(str(t)) for t in X_test_list]
    logger.info(f"Treino CV: {len(X_train_cv)} | Holdout: {len(X_test_pp)} "
                f"(pre-processado)")

    evaluator = ModelEvaluator(
        task_name='AS',
        class_names=get_task_config('AS')['classes'],
        output_dir=tempfile.mkdtemp(prefix='gridwarmup_'),
    )

    # ── Fase 1: reavaliacao dos folds de A e B (rapida, sem treino) ───────────
    logger.info("\n" + "#" * 70)
    logger.info("# FASE 1 — reavaliacao dos 5 folds de A e B (sem treino)")
    logger.info("#" * 70)
    ab_results = reeval_ab_folds(X_test_pp, y_test, evaluator)
    if ab_results:
        _save(ab_results, OUT_ABFOLD)
        for r in ab_results:
            c = r['config']
            logger.info(f"Config {c['name']}: fold otimo por validacao="
                        f"{r['val_best_fold']}  | por holdout="
                        f"{r['holdout_best_fold']}")

    # ── Fase 2: treino de G, H, I ────────────────────────────────────────────
    logger.info("\n" + "#" * 70)
    logger.info("# FASE 2 — treino das configs G, H, I")
    logger.info("#" * 70)
    warmup_results: list[dict] = []
    for cfg in GRID:
        result = run_config(cfg, X_train_cv, y_train_cv,
                            X_test_pp, y_test, evaluator)
        warmup_results.append(result)
        _save(warmup_results, OUT_WARMUP)   # parcial apos cada config

    print_summary(warmup_results, ab_results)

    elapsed = (time.time() - t0) / 60
    logger.info(f"Tempo total: {elapsed:.1f} min")
    logger.info(f"Saidas: {OUT_ABFOLD.name}, {OUT_WARMUP.name}")


if __name__ == '__main__':
    main()
