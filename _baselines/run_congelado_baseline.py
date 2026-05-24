"""
Re-executa o baseline "BERTimbau congelado + Regressao Logistica" no holdout,
em tres variantes de pre-processamento, para esclarecer a discrepancia de
numeros (artigo: 74,73%; run_all_02.py em texto cru: 73,82%).

BERTimbau congelado = modelo pre-treinado usado apenas como extrator de
embeddings (mean pooling), alimentando uma Regressao Logistica. Sem fine-tuning.

Variantes de pre-processamento aplicadas aos textos antes do embedding:
  A. texto cru
  B. limpeza apenas
  C. limpeza + marcacao de negacoes (preprocess_for_sentiment completo)

NAO altera o artigo nem nenhum resultado existente. Saida nova:
  src/_baselines/congelado_baseline_results.json

Execucao:
  cd <projeto>/src
  python -m _baselines.run_congelado_baseline
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # src/_baselines/
SRC_DIR    = SCRIPT_DIR.parent                    # src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.nlp.config import PATHS

CORPUS   = SRC_DIR / "app" / "nlp" / "datasets" / "corpus.csv"
OUT_FILE = SCRIPT_DIR / "congelado_baseline_results.json"
SEED     = 42
LABELS   = ['Negativo', 'Neutro', 'Positivo']
MODEL    = "neuralmind/bert-base-portuguese-cased"

# valor reportado no artigo (Tabela de baselines)
ARTIGO = {'accuracy': 0.7473, 'f1_macro': 0.7481}


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


def main() -> None:
    PATHS['corpus_file'] = str(CORPUS)

    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, classification_report,
                                 f1_score)
    from transformers import AutoModel, AutoTokenizer

    from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test

    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test()
    X_train = [str(x) for x in X_train]
    X_test  = [str(x) for x in X_test]
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test,  dtype=int)
    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    tok   = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    def embed(texts: list, batch_size: int = 16, max_len: int = 128) -> np.ndarray:
        out_all = []
        n = len(texts)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = texts[i:i + batch_size]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt").to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                out_all.append(emb.cpu().numpy())
        return np.vstack(out_all)

    variants = {
        'A. texto cru':
            (lambda t: t),
        'B. limpeza apenas':
            (lambda t: _clean_text(t)),
        'C. limpeza + negacoes (completo)':
            (lambda t: _handle_negations(_clean_text(t))),
    }

    results = []
    for name, fn in variants.items():
        print(f"\n>>> Variante {name}")
        Xtr = embed([fn(t) for t in X_train])
        Xte = embed([fn(t) for t in X_test])
        clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                                 random_state=SEED)
        clf.fit(Xtr, y_train)
        y_pred = clf.predict(Xte)

        acc  = float(accuracy_score(y_test, y_pred))
        f1m  = float(f1_score(y_test, y_pred, average='macro'))
        f1pc = [float(x) for x in f1_score(y_test, y_pred, average=None)]
        print(f"    accuracy={acc*100:.2f}%  f1_macro={f1m*100:.2f}%  "
              f"(Neg={f1pc[0]*100:.1f}  Neu={f1pc[1]*100:.1f}  "
              f"Pos={f1pc[2]*100:.1f})")
        print(classification_report(y_test, y_pred, target_names=LABELS,
                                    digits=4, zero_division=0))
        results.append({
            'variante':     name,
            'accuracy':     acc,
            'f1_macro':     f1m,
            'f1_per_class': f1pc,
        })

    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos -> {OUT_FILE.name}")

    print("\n" + "=" * 74)
    print("RESUMO — BERTimbau congelado + Regressao Logistica")
    print(f"{'Variante':<36} {'Acuracia':>10} {'F1 Macro':>10}")
    print("-" * 74)
    print(f"{'Artigo (valor reportado)':<36} "
          f"{ARTIGO['accuracy']*100:>9.2f}% {ARTIGO['f1_macro']*100:>9.2f}%")
    for r in results:
        print(f"{r['variante']:<36} "
              f"{r['accuracy']*100:>9.2f}% {r['f1_macro']*100:>9.2f}%")
    print("=" * 74)


if __name__ == '__main__':
    main()
