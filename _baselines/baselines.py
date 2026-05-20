"""
Baselines para o artigo WICS 2026.

Compara tres baselines com o modelo BERTimbau ajustado:
  1. Lexico simples de polaridade em PT-BR (palavras positivas/negativas).
  2. TF-IDF + Regressao Logistica.
  3. BERTimbau congelado (sem fine-tuning) + Regressao Logistica
     (interpretacao pratica de "zero-shot" sobre features pre-treinadas).

Split: 80/20 estratificado, semente fixa (random_state=42).
Saida: stdout com tabela de metricas globais e por classe.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
CORPUS = ROOT / "corpus.csv"
SEED = 42
LABELS = ["Negativo", "Neutro", "Positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


def load_corpus() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(CORPUS, sep=";", encoding="utf-8", engine="python",
                     on_bad_lines="skip")
    df = df.dropna(subset=["FRASE", "AS"]).copy()
    df["FRASE"] = (df["FRASE"].astype(str)
                   .str.strip()
                   .str.strip('"')
                   .str.strip())
    df["AS"] = df["AS"].astype(str).str.strip()
    df = df[df["AS"].isin(LABELS)]
    df = df[df["FRASE"].str.len() > 0]
    df["y"] = df["AS"].map(LABEL_TO_ID).astype(int)
    print(f"Corpus carregado: {len(df)} frases")
    print("Distribuicao por classe:")
    for l in LABELS:
        n = int((df["AS"] == l).sum())
        print(f"  {l:10s}: {n}")
    return df["FRASE"].values, df["y"].values


# ------------------------------------------------------------
# Baseline 1: lexico simples PT-BR
# ------------------------------------------------------------

NEG_WORDS = {
    "triste", "tristeza", "raiva", "raivoso", "irritado", "irritacao",
    "medo", "medos", "amedrontado", "assustado", "assustador", "aterrorizado",
    "ruim", "horrivel", "pessimo", "horror", "terror", "terrivel",
    "odio", "odeio", "odiar", "detesto", "detestar",
    "mau", "ma", "maus", "mas", "feio", "feia",
    "perigo", "perigoso", "perigosa", "morrer", "morte", "morto", "matar", "morta",
    "sangue", "sangrento", "dor", "dolorido", "sofrer", "sofrimento",
    "dificil", "chorar", "chorando", "machucar", "machucado", "ferir", "ferido",
    "culpa", "culpado", "desespero", "desesperado",
    "idiota", "burro", "tonto", "estupido", "imbecil",
    "frustracao", "frustrado", "derrota", "perder", "perdedor", "fracasso",
    "agressao", "agressivo", "violencia", "violento", "arma",
    "assassino", "assassinato", "vinganca", "vingativo",
    "traicao", "traidor", "mentira", "mentiroso", "falso",
    "engano", "enganar", "enganador", "pesadelo",
    "covarde", "solitario", "solidao", "abandonado", "desistir",
    "deprimido", "depressao", "ansiedade", "ansioso", "panico",
    "nojento", "nojo", "asco", "desagradavel", "desprezo", "desprezivel",
    "desumano", "cruel", "crueldade", "sadico",
    "tortura", "torturar", "torturado", "vitima",
    "merda", "porra", "caralho", "puta", "putaria", "fdp", "buceta",
    "cu", "babaca", "otario", "viado", "bicha", "veado",
    "vagabundo", "vagabunda", "ladrao", "ladra",
    "diabo", "demonio", "infernal", "inferno",
    "doente", "doenca", "vomito", "vomitar", "vomitando",
    "lixo", "porcaria", "porco",
    "guerra", "bomba", "tiro", "tiroteio", "atentado",
    "estupro", "estuprador", "abuso", "abusivo",
}

POS_WORDS = {
    "feliz", "felicidade", "alegria", "alegre", "alegremente",
    "amor", "amar", "amado", "amada", "amoroso",
    "carinho", "carinhoso", "carinhosa", "beijo", "abraco", "abracar",
    "sorriso", "sorrir", "sorridente", "rir", "rindo",
    "divertido", "diversao", "divertir", "brincar", "brincadeira",
    "jogo", "jogar", "jogador", "amigo", "amiga", "amizade",
    "vencer", "vencedor", "vitoria", "sucesso", "conquista", "conquistar",
    "realizar", "realizacao", "realizado",
    "aprender", "aprendizado", "aprendizagem", "ensinar", "ensino",
    "professor", "aluno", "estudante", "estudar", "escola",
    "educacao", "educado", "educativo", "educacional",
    "companheiro", "companhia", "familia", "familiar",
    "pai", "mae", "filho", "filha", "irmao", "irma", "irmaos",
    "contente", "satisfeito", "satisfacao", "grato", "gratidao",
    "obrigado", "obrigada", "agradecer", "agradecido",
    "elogio", "elogiar", "elogiado",
    "admirar", "admiracao", "admiravel",
    "maravilha", "maravilhoso", "maravilhosa", "incrivel",
    "fantastico", "espetacular", "sensacional", "excelente",
    "otimo", "otima", "bom", "boa", "melhor",
    "perfeito", "perfeita", "perfeicao",
    "bonito", "bonita", "linda", "lindo", "beleza", "belo", "bela",
    "encantador", "encantar", "encantado",
    "fascinante", "fascinar", "fascinado",
    "interessante", "curioso", "curiosidade",
    "descobrir", "descoberta", "aventura", "aventureiro",
    "heroi", "heroico", "salvar", "salvador",
    "protegido", "proteger", "protecao", "protetor",
    "cuidar", "cuidado", "cuidadoso", "gentil", "gentileza",
    "bondade", "bondoso", "generoso", "generosidade",
    "compartilhar", "ajudar", "ajuda", "colaborar",
    "solidario", "solidariedade", "paz", "pacifico",
    "tranquilo", "tranquilidade", "calmo", "sereno", "serenidade",
    "esperanca", "esperancoso", "otimista", "otimismo",
    "positivo", "luz", "iluminar", "brilho", "brilhar", "brilhante",
    "parabens", "viva", "uhuu", "obrigadinho",
}


def lexicon_predict(sentences: np.ndarray) -> np.ndarray:
    preds = []
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        pos = sum(1 for t in toks if t in POS_WORDS)
        neg = sum(1 for t in toks if t in NEG_WORDS)
        if neg > pos:
            preds.append(0)
        elif pos > neg:
            preds.append(2)
        else:
            preds.append(1)
    return np.array(preds)


# ------------------------------------------------------------
# Baseline 2: TF-IDF + Regressao Logistica
# ------------------------------------------------------------

def run_tfidf(X_train, y_train, X_test):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True,
                          sublinear_tf=True)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced",
                             random_state=SEED)
    clf.fit(Xtr, y_train)
    return clf.predict(Xte)


# ------------------------------------------------------------
# Baseline 3: BERTimbau congelado + Regressao Logistica
# ------------------------------------------------------------

def run_bertimbau_frozen(X_train, y_train, X_test):
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    name = "neuralmind/bert-base-portuguese-cased"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()

    def embed(texts, batch_size=16, max_len=128):
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
    return clf.predict(Xte)


# ------------------------------------------------------------
# Relatorio
# ------------------------------------------------------------

def metrics_block(name: str, y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    pw = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rw = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    per_class = {}
    for i, l in enumerate(LABELS):
        p = precision_score(y_true, y_pred, labels=[i], average="macro",
                            zero_division=0)
        r = recall_score(y_true, y_pred, labels=[i], average="macro",
                         zero_division=0)
        f = f1_score(y_true, y_pred, labels=[i], average="macro",
                     zero_division=0)
        per_class[l] = {"precision": p, "recall": r, "f1": f}

    print(f"\n=== {name} ===")
    print(f"  Acuracia      : {acc * 100:.2f}%")
    print(f"  F1 Macro      : {f1m * 100:.2f}%")
    print(f"  F1 Weighted   : {f1w * 100:.2f}%")
    print(f"  Precisao W.   : {pw * 100:.2f}%")
    print(f"  Recall W.     : {rw * 100:.2f}%")
    print("  Por classe (precisao / recall / F1):")
    for l, m in per_class.items():
        print(f"    {l:10s}  {m['precision']*100:6.2f} / "
              f"{m['recall']*100:6.2f} / {m['f1']*100:6.2f}")
    print(classification_report(y_true, y_pred, target_names=LABELS,
                                digits=4, zero_division=0))
    return {
        "name": name,
        "accuracy": acc,
        "f1_macro": f1m,
        "f1_weighted": f1w,
        "precision_weighted": pw,
        "recall_weighted": rw,
        "per_class": per_class,
    }


def main():
    t0 = time.time()
    X, y = load_corpus()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    print(f"\nSplit 80/20 estratificado (seed={SEED}): "
          f"treino={len(X_train)}, teste={len(X_test)}")

    results = []

    print("\n>>> Baseline 1: Lexico simples PT-BR")
    y_lex = lexicon_predict(X_test)
    results.append(metrics_block("Lexico simples (PT-BR)", y_test, y_lex))

    print("\n>>> Baseline 2: TF-IDF + Regressao Logistica")
    y_tfidf = run_tfidf(X_train, y_train, X_test)
    results.append(metrics_block("TF-IDF + Regressao Logistica",
                                 y_test, y_tfidf))

    print("\n>>> Baseline 3: BERTimbau congelado + Regressao Logistica")
    try:
        y_bert = run_bertimbau_frozen(X_train, y_train, X_test)
        results.append(metrics_block(
            "BERTimbau congelado + Regressao Logistica (zero-shot)",
            y_test, y_bert))
    except Exception as e:
        print(f"  Falha no baseline BERTimbau: {e}")
        import traceback
        traceback.print_exc()

    out = ROOT / "baselines_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados salvos em {out}")
    print(f"Tempo total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
