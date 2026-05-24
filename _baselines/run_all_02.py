"""
Re-executa todos os baselines + BERTimbau ajustado no holdout oficial de 550 amostras.

Usa get_data_for_cv_and_test() — mesma funcao usada por evaluate_sentiment.py —
para garantir exatamente o mesmo split do holdout que gerou holdout_oficial.json.

Saida (sem sobrescrever arquivos originais):
  baseline_sentilexpt_02.json
  baseline_tfidf_02.json
  baseline_bertimbau_02.json        (congelado + LR)
  baseline_bertimbau_ajustado_02.json
  baselines_results_02.json         (todos consolidados)

Execucao:
  cd <projeto>/src
  python -m _baselines.run_all_02
  ou:
  python _baselines/run_all_02.py
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent   # src/_baselines/
SRC_DIR    = SCRIPT_DIR.parent                 # src/
OUT_DIR    = SCRIPT_DIR

# Adiciona src/ ao path para poder importar app.*
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)

SEED       = 42
LABELS     = ["Negativo", "Neutro", "Positivo"]
SENTILEXPT = SCRIPT_DIR / "SentiLex-flex-PT02.txt"
MODEL_PATH = (SRC_DIR / "app" / "nlp" / "models" / "trained"
              / "AS_sentiment_cv_fold_1_smote_20260515_193838")


# ── data ──────────────────────────────────────────────────────────────────────
def load_holdout():
    """Mesma divisao 80/20 usada por evaluate_sentiment.py (seed=42, stratify)."""
    from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
    (X_train, y_train), (X_test, y_test) = get_data_for_cv_and_test()
    X_train = np.array(X_train, dtype=object)
    X_test  = np.array(X_test,  dtype=object)
    y_train = np.array(y_train, dtype=int)
    y_test  = np.array(y_test,  dtype=int)
    print(f"\nCorpus dividido: treino={len(X_train)}, teste={len(X_test)}")
    dist = dict(sorted(Counter(y_test).items()))
    dist_named = {LABELS[k]: v for k, v in dist.items()}
    print(f"  Distribuicao teste: {dist_named}")
    return X_train, y_train, X_test, y_test


# ── metricas ──────────────────────────────────────────────────────────────────
def build_result(name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    pw  = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rw  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    per_class = {}
    for i, label in enumerate(LABELS):
        p = precision_score(y_true, y_pred, labels=[i], average="macro", zero_division=0)
        r = recall_score(y_true, y_pred,    labels=[i], average="macro", zero_division=0)
        f = f1_score(y_true, y_pred,        labels=[i], average="macro", zero_division=0)
        per_class[label] = {"precision": p, "recall": r, "f1": f}
    print(f"\n=== {name} ===")
    print(f"  Acuracia : {acc*100:.2f}%  |  F1 Macro : {f1m*100:.2f}%  |  F1 Weighted : {f1w*100:.2f}%")
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=4, zero_division=0))
    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "precision_weighted": pw,
        "recall_weighted": rw,
        "per_class": per_class,
    }


def save_json(data: dict | list, filename: str) -> None:
    path = OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Salvo -> {path.name}")


# ── 1. SentiLex-PT ────────────────────────────────────────────────────────────
def load_sentilexpt(path: Path) -> dict[str, int]:
    pol_re = re.compile(r"POL:N0=(-?1|0)")
    votes: dict[str, list[int]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            comma = line.find(",")
            if comma == -1:
                continue
            word = line[:comma].lower()
            m = pol_re.search(line)
            if not m:
                continue
            pol = int(m.group(1))
            if pol != 0:
                votes[word].append(pol)
    result: dict[str, int] = {}
    for word, pols in votes.items():
        pos, neg = pols.count(1), pols.count(-1)
        if pos > neg:
            result[word] = 1
        elif neg > pos:
            result[word] = -1
    return result


def sentilexpt_predict(texts: np.ndarray, pol_dict: dict[str, int]) -> np.ndarray:
    preds = []
    for s in texts:
        toks = re.findall(r"\w+", s.lower())
        pos = sum(1 for t in toks if pol_dict.get(t, 0) > 0)
        neg = sum(1 for t in toks if pol_dict.get(t, 0) < 0)
        preds.append(0 if neg > pos else 2 if pos > neg else 1)
    return np.array(preds)


def run_sentilexpt(X_test: np.ndarray, y_test: np.ndarray) -> dict | None:
    if not SENTILEXPT.exists():
        print(f"  AVISO: {SENTILEXPT} nao encontrado — baseline ignorado.")
        return None
    pol_dict = load_sentilexpt(SENTILEXPT)
    print(f"  SentiLex: {len(pol_dict)} formas "
          f"({sum(1 for v in pol_dict.values() if v > 0)} pos, "
          f"{sum(1 for v in pol_dict.values() if v < 0)} neg)")
    y_pred = sentilexpt_predict(X_test, pol_dict)
    return build_result("SentiLex-PT (flex, POL:N0)", y_test, y_pred)


# ── 2. TF-IDF + Regressao Logistica ──────────────────────────────────────────
def run_tfidf(X_train: np.ndarray, y_train: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray) -> dict:
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                          lowercase=True, sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    return build_result("TF-IDF + Regressao Logistica", y_test, y_pred)


# ── 3. BERTimbau congelado + Regressao Logistica ─────────────────────────────
def run_bertimbau_frozen(X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    name = "neuralmind/bert-base-portuguese-cased"
    tok   = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed(texts: np.ndarray, batch_size: int = 16, max_len: int = 128) -> np.ndarray:
        out_all = []
        n = len(texts)
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = [str(x) for x in texts[i:i + batch_size]]
                enc = tok(batch, padding=True, truncation=True,
                          max_length=max_len, return_tensors="pt").to(device)
                out = model(**enc)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
                out_all.append(emb.cpu().numpy())
                if (i // batch_size) % 5 == 0:
                    print(f"    embed {i + len(batch)}/{n}", flush=True)
        return np.vstack(out_all)

    print("  Embedding treino...")
    Xtr = embed(X_train)
    print("  Embedding teste...")
    Xte = embed(X_test)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    return build_result(
        "BERTimbau congelado + Regressao Logistica (zero-shot)", y_test, y_pred)


# ── 4. BERTimbau ajustado (proposto) ─────────────────────────────────────────
def run_bertimbau_ajustado(X_test: np.ndarray, y_test: np.ndarray) -> dict:
    import torch
    from transformers import BertForSequenceClassification, BertTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    print(f"  Modelo: {MODEL_PATH.name}")
    tok   = BertTokenizer.from_pretrained(str(MODEL_PATH))
    model = BertForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device).eval()

    preds_all: list[int] = []
    batch_size = 32
    n = len(X_test)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = [str(x) for x in X_test[i:i + batch_size]]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=128, return_tensors="pt").to(device)
            logits = model(**enc).logits
            preds_all.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            if (i // batch_size) % 5 == 0:
                print(f"    predict {i + len(batch)}/{n}", flush=True)
    y_pred = np.array(preds_all)
    return build_result("BERTimbau ajustado (proposto)", y_test, y_pred)


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    t0 = time.time()
    X_train, y_train, X_test, y_test = load_holdout()
    all_results: list[dict] = []

    # 1. SentiLex-PT
    print("\n" + "=" * 60)
    print(">>> 1/4  SentiLex-PT (flex, POL:N0)")
    r = run_sentilexpt(X_test, y_test)
    if r:
        save_json(r, "baseline_sentilexpt_02.json")
        all_results.append(r)

    # 2. TF-IDF + LR
    print("\n" + "=" * 60)
    print(">>> 2/4  TF-IDF + Regressao Logistica")
    r = run_tfidf(X_train, y_train, X_test, y_test)
    save_json(r, "baseline_tfidf_02.json")
    all_results.append(r)

    # 3. BERTimbau congelado + LR
    print("\n" + "=" * 60)
    print(">>> 3/4  BERTimbau congelado + Regressao Logistica (zero-shot)")
    try:
        r = run_bertimbau_frozen(X_train, y_train, X_test, y_test)
        save_json(r, "baseline_bertimbau_02.json")
        all_results.append(r)
    except Exception as exc:
        print(f"  Falha: {exc}")
        import traceback
        traceback.print_exc()

    # 4. BERTimbau ajustado
    print("\n" + "=" * 60)
    print(">>> 4/4  BERTimbau ajustado (proposto)")
    try:
        r = run_bertimbau_ajustado(X_test, y_test)
        save_json(r, "baseline_bertimbau_ajustado_02.json")
        all_results.append(r)
    except Exception as exc:
        print(f"  Falha: {exc}")
        import traceback
        traceback.print_exc()

    # Consolidado
    if all_results:
        save_json(all_results, "baselines_results_02.json")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Concluido em {elapsed:.1f}s — arquivos salvos em {OUT_DIR}")
    print("=" * 60)

    # Resumo comparativo
    print("\n### RESUMO COMPARATIVO ###")
    header = f"{'Metodo':<48} {'Acuracia':>9} {'F1 Macro':>9} {'F1 Neg':>7} {'F1 Neu':>7} {'F1 Pos':>7}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        pc = r["per_class"]
        print(f"{r['name']:<48} "
              f"{r['accuracy']*100:>8.2f}% "
              f"{r['f1_macro']*100:>8.2f}% "
              f"{pc['Negativo']['f1']*100:>6.1f}% "
              f"{pc['Neutro']['f1']*100:>6.1f}% "
              f"{pc['Positivo']['f1']*100:>6.1f}%")


if __name__ == "__main__":
    main()
