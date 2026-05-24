"""
Ensemble dos 5 folds — reducao de variancia.

A reavaliacao dos folds (ab_folds_reeval.json) mostrou dispersao de ~3,5pp no
holdout: um modelo de fold unico e um estimador ruidoso. O ensemble faz a media
das probabilidades (softmax) dos 5 modelos de uma config — metodo definido a
priori (NAO e holdout-fishing): reduz a variancia e costuma entregar um numero
estavel proximo da media do CV, as vezes um pouco acima.

Avalia o ensemble de A (3e-5/wr100/ep5) e de B (3e-5/wr500/ep5) no holdout
pre-processado (limpeza + negacoes — metodo que reproduz o artigo). Compara com
o alvo do artigo (77,27% / 77,49%) e com os folds individuais.

Saida: src/_baselines/ensemble_results.json  (arquivo novo)

Execucao:
  cd <projeto>/src
  python -m _baselines.ensemble_folds
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # src/_baselines/
SRC_DIR    = SCRIPT_DIR.parent                    # src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.nlp.config import PATHS, get_task_config

CORPUS   = SRC_DIR / "app" / "nlp" / "datasets" / "corpus.csv"
GRID_3E5 = SCRIPT_DIR / "grid_3e5_results.json"   # contem A, B, C
OUT_FILE = SCRIPT_DIR / "ensemble_results.json"

TARGET = {'accuracy': 0.7727272727272727, 'f1_macro': 0.7748822217541185}
LABELS = ['Negativo', 'Neutro', 'Positivo']


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
    return _handle_negations(_clean_text(text))


def main() -> None:
    print("=" * 84)
    print("ENSEMBLE DOS 5 FOLDS  (media das probabilidades, holdout pre-processado)")
    print(f"Alvo do artigo: acc={TARGET['accuracy']*100:.2f}%  "
          f"f1_macro={TARGET['f1_macro']*100:.2f}%")
    print("=" * 84)

    PATHS['corpus_file'] = str(CORPUS)

    import numpy as np
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from transformers import BertForSequenceClassification, BertTokenizer

    from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
    from app.nlp.evaluation.model_evaluator import ModelEvaluator

    _, (test_texts, test_labels) = get_data_for_cv_and_test()
    y_test = np.array([int(y) for y in test_labels])
    X_pp   = [preprocess(str(t)) for t in test_texts]
    print(f"Holdout: {len(y_test)} amostras (pre-processadas)\n")

    evaluator = ModelEvaluator(
        task_name='AS',
        class_names=get_task_config('AS')['classes'],
        output_dir=tempfile.mkdtemp(prefix='ensemble_'),
    )

    with open(GRID_3E5, encoding='utf-8') as f:
        grid = json.load(f)

    results: list[dict] = []
    for name in ('A', 'B'):
        entry = next((x for x in grid if x['config']['name'] == name), None)
        if entry is None:
            print(f"AVISO: Config {name} nao encontrada — ignorada.")
            continue
        cfg = entry['config']
        print(f">>> Config {name}  (lr={cfg['lr']} / wr={cfg['warmup']} / ep=5)")

        probs_per_fold: list = []
        fold_f1s:       list = []
        for fd in entry['folds']:
            mp = Path(fd['model_path'])
            if not mp.exists():
                print(f"    AVISO: fold {fd['fold']} ausente — {mp}")
                continue
            model     = BertForSequenceClassification.from_pretrained(str(mp))
            tokenizer = BertTokenizer.from_pretrained(str(mp))
            res = evaluator.evaluate_model(
                model=model, tokenizer=tokenizer,
                test_texts=X_pp, test_labels=list(y_test),
                max_length=128, batch_size=32,
            )
            probs_per_fold.append(np.array(res['probabilities']))
            fold_f1s.append(float(res['f1_macro']))
            print(f"    fold {fd['fold']}: holdout f1_macro={res['f1_macro']*100:.2f}%")

        if not probs_per_fold:
            continue

        # soft voting: media das probabilidades dos folds
        ens_probs = np.mean(probs_per_fold, axis=0)
        y_pred    = ens_probs.argmax(axis=1)

        acc  = float(accuracy_score(y_test, y_pred))
        f1m  = float(f1_score(y_test, y_pred, average='macro'))
        f1w  = float(f1_score(y_test, y_pred, average='weighted'))
        f1pc = [float(x) for x in f1_score(y_test, y_pred, average=None)]
        cm   = confusion_matrix(y_test, y_pred).tolist()

        fold_mean = float(np.mean(fold_f1s))
        fold_std  = float(np.std(fold_f1s))
        fold_best = float(np.max(fold_f1s))

        print(f"    -- folds: media={fold_mean*100:.2f}%  "
              f"std={fold_std*100:.2f}  melhor={fold_best*100:.2f}%")
        print(f"    == ENSEMBLE: acc={acc*100:.2f}%  f1_macro={f1m*100:.2f}%  "
              f"(Neg={f1pc[0]*100:.1f}  Neu={f1pc[1]*100:.1f}  Pos={f1pc[2]*100:.1f})")
        d_acc = (acc - TARGET['accuracy']) * 100
        d_f1  = (f1m - TARGET['f1_macro']) * 100
        print(f"       vs artigo: {d_acc:+.2f}pp acc | {d_f1:+.2f}pp f1_macro")
        print(classification_report(y_test, y_pred, target_names=LABELS,
                                    digits=4, zero_division=0))

        results.append({
            'config':         cfg,
            'fold_f1_macros': fold_f1s,
            'fold_mean_f1':   fold_mean,
            'fold_std_f1':    fold_std,
            'fold_best_f1':   fold_best,
            'ensemble': {
                'accuracy':         acc,
                'f1_macro':         f1m,
                'f1_weighted':      f1w,
                'f1_per_class':     f1pc,
                'confusion_matrix': cm,
            },
        })

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Resultados salvos -> {OUT_FILE.name}")

    # ── resumo ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 84)
    print("RESUMO")
    print("=" * 84)
    print(f"{'Modelo':<32} {'Acc':>9} {'F1 Macro':>10}"
          f" {'F1 Neg':>8} {'F1 Neu':>8} {'F1 Pos':>8}")
    print("-" * 84)
    print(f"{'Referencia (artigo)':<32} {TARGET['accuracy']*100:>8.2f}%"
          f" {TARGET['f1_macro']*100:>9.2f}%"
          f" {91.81:>7.1f}% {72.73:>7.1f}% {67.92:>7.1f}%")
    for r in results:
        e = r['ensemble']
        pc = e['f1_per_class']
        print(f"{'Ensemble Config '+r['config']['name']+' (5 folds)':<32}"
              f" {e['accuracy']*100:>8.2f}%"
              f" {e['f1_macro']*100:>9.2f}%"
              f" {pc[0]*100:>7.1f}% {pc[1]*100:>7.1f}% {pc[2]*100:>7.1f}%")
    print("=" * 84)


if __name__ == '__main__':
    main()
