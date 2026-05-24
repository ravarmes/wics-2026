"""
Reavaliacao CORRETA dos modelos do grid (A-F) no holdout.

Motivo:
  verify_article_corpus.py provou que o modelo so atinge o desempenho real
  quando avaliado com pre-processamento (limpeza + marcacao de negacoes) —
  a mesma transformacao usada no treino. Avaliar em texto cru subestima em
  ~2,4pp. Os scripts de grid (train_grid_3e5.py / train_grid_def.py) avaliaram
  o holdout em texto CRU; logo, as metricas de holdout de A-F estao subestimadas.

Este script:
  - Reconstroi o holdout (corpus atual, split 80/20, seed=42).
  - Aplica preprocess_for_sentiment (Variante C — reproduz o 77,27% do artigo
    digito a digito).
  - Reavalia o melhor fold de cada config A-F + o modelo de referencia do artigo.
  - Compara cada um com o alvo do artigo (77,27% acc / 77,49% f1_macro).

Nao altera nenhum arquivo existente. Saida nova:
  src/_baselines/grid_reeval_results.json

Execucao:
  cd <projeto>/src
  python -m _baselines.reeval_grid_holdout
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

REF_MODEL  = (SRC_DIR / "app" / "nlp" / "models" / "trained"
              / "AS_sentiment_cv_fold_1_smote_20260515_193838")
CORPUS     = SRC_DIR / "app" / "nlp" / "datasets" / "corpus.csv"
GRID_FILES = [SCRIPT_DIR / "grid_3e5_results.json",
              SCRIPT_DIR / "grid_def_results.json"]
OUT_FILE   = SCRIPT_DIR / "grid_reeval_results.json"

# Alvo do artigo (holdout_oficial.json)
TARGET = {'accuracy': 0.7727272727272727, 'f1_macro': 0.7748822217541185}


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


def main() -> None:
    print("=" * 92)
    print("REAVALIACAO CORRETA DOS MODELOS DO GRID (holdout com pre-processamento)")
    print(f"Alvo do artigo: acc={TARGET['accuracy']*100:.2f}%  "
          f"f1_macro={TARGET['f1_macro']*100:.2f}%")
    print("=" * 92)

    PATHS['corpus_file'] = str(CORPUS)

    from transformers import BertForSequenceClassification, BertTokenizer

    from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
    from app.nlp.evaluation.model_evaluator import ModelEvaluator

    _, (test_texts, test_labels) = get_data_for_cv_and_test()
    test_texts_pp = [preprocess(t) for t in test_texts]
    print(f"Holdout: {len(test_labels)} amostras (pre-processadas)\n")

    # ── monta a lista de modelos a reavaliar ──────────────────────────────────
    targets: list[dict] = [{
        'name':        'Referencia (artigo)',
        'config':      {'name': 'REF', 'lr': 5e-5, 'warmup': 500,
                        'epochs': 5, 'extra': 'modelo do artigo'},
        'model_path':  str(REF_MODEL),
        'holdout_raw': None,
    }]
    for gf in GRID_FILES:
        if not gf.exists():
            print(f"AVISO: {gf.name} nao encontrado — ignorado.")
            continue
        with open(gf, encoding='utf-8') as f:
            for r in json.load(f):
                targets.append({
                    'name':        f"Config {r['config']['name']}",
                    'config':      r['config'],
                    'model_path':  r['best_model_path'],
                    'holdout_raw': r['holdout'],
                })

    evaluator = ModelEvaluator(
        task_name='AS',
        class_names=get_task_config('AS')['classes'],
        output_dir=tempfile.mkdtemp(prefix='reeval_'),
    )

    # ── reavalia cada modelo ──────────────────────────────────────────────────
    results: list[dict] = []
    for t in targets:
        mp = Path(t['model_path'])
        if not mp.exists():
            print(f"AVISO: modelo nao encontrado — {mp}")
            continue

        model     = BertForSequenceClassification.from_pretrained(str(mp))
        tokenizer = BertTokenizer.from_pretrained(str(mp))
        res = evaluator.evaluate_model(
            model=model, tokenizer=tokenizer,
            test_texts=test_texts_pp, test_labels=test_labels,
            max_length=128, batch_size=32,
        )
        results.append({
            'name':        t['name'],
            'config':      t['config'],
            'model_path':  t['model_path'],
            'holdout_raw': t['holdout_raw'],
            'holdout_correto': {
                'accuracy':     res['accuracy'],
                'f1_macro':     res['f1_macro'],
                'f1_weighted':  res['f1_weighted'],
                'f1_per_class': res['f1_per_class'],
                'confusion_matrix': res['confusion_matrix'],
            },
        })

    # ── salva JSON ────────────────────────────────────────────────────────────
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResultados salvos -> {OUT_FILE.name}")

    # ── tabela comparativa ────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("RESUMO  —  holdout cru (subestimado)  vs  holdout correto")
    print("=" * 92)
    header = (f"{'Config':<34} {'F1 cru':>8} {'F1 correto':>11}"
              f" {'ganho':>7} {'Acc correto':>12} {'vs 77,27%':>10}")
    print(header)
    print("-" * 92)
    for r in results:
        c   = r['config']
        hc  = r['holdout_correto']
        raw = r['holdout_raw']
        label = (f"{r['name']} (lr={c['lr']}/wr={c['warmup']}"
                 f"/ep={c['epochs']})")
        f1_raw_s  = f"{raw['f1_macro']*100:.2f}%" if raw else "  -  "
        gain_s    = (f"{(hc['f1_macro']-raw['f1_macro'])*100:+.2f}"
                     if raw else "  -  ")
        d_target  = (hc['accuracy'] - TARGET['accuracy']) * 100
        flag = "BATE/SUPERA" if d_target >= -0.005 else f"{d_target:+.2f}pp"
        print(f"{label:<34} {f1_raw_s:>8} {hc['f1_macro']*100:>10.2f}%"
              f" {gain_s:>7} {hc['accuracy']*100:>11.2f}%"
              f" {flag:>10}")
    print("=" * 92)
    print("F1 cru = holdout em texto cru (scripts de grid). "
          "F1 correto = holdout com pre-processamento.")


if __name__ == '__main__':
    main()
