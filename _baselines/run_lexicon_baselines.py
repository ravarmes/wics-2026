"""Roda apenas os baselines léxicos (Léxico simples + SentiLex-PT) no holdout."""
import re
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, classification_report)

ROOT = Path(__file__).parent
SENTILEXPT = ROOT / "SentiLex-flex-PT02.txt"
SEED = 42
LABELS = ["Negativo", "Neutro", "Positivo"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}

# ---- corpus ----
candidates = [
    ROOT / "corpus.csv",
    ROOT.parent / "app" / "nlp" / "datasets" / "corpus.csv",
    ROOT.parent / "data" / "corpus.csv",
]
CORPUS = next((p for p in candidates if p.exists()), candidates[0])
print("Corpus:", CORPUS)

df = pd.read_csv(CORPUS, sep=";", encoding="utf-8", engine="python",
                 on_bad_lines="skip")
df = df.dropna(subset=["FRASE", "AS"]).copy()
df["FRASE"] = df["FRASE"].astype(str).str.strip().str.strip('"').str.strip()
df["AS"] = df["AS"].astype(str).str.strip()
df = df[df["AS"].isin(LABELS)]
df["y"] = df["AS"].map(LABEL_TO_ID).astype(int)
print(f"Corpus: {len(df)} frases")
for l in LABELS:
    print(f"  {l}: {(df['AS'] == l).sum()}")

X, y = df["FRASE"].values, df["y"].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"Split: treino={len(X_train)}, teste={len(X_test)}")


def report(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    pw = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rw = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    pc = {}
    for i, l in enumerate(LABELS):
        pc[l] = {
            "precision": float(precision_score(y_true, y_pred, labels=[i],
                                               average="macro", zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, labels=[i],
                                            average="macro", zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, labels=[i],
                                        average="macro", zero_division=0)),
        }
    print(f"\n=== {name} ===")
    print(f"  Acuracia: {acc*100:.2f}%  F1-macro: {f1m*100:.2f}%  "
          f"F1-weighted: {f1w*100:.2f}%")
    for l, m in pc.items():
        print(f"  {l}: P={m['precision']*100:.2f}  R={m['recall']*100:.2f}  "
              f"F1={m['f1']*100:.2f}")
    print(classification_report(y_true, y_pred, target_names=LABELS,
                                digits=4, zero_division=0))
    return {
        "name": name,
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "precision_weighted": float(pw),
        "recall_weighted": float(rw),
        "per_class": pc,
    }


# ---- Baseline 1: Léxico simples ----
NEG_WORDS = {
    "triste", "tristeza", "raiva", "raivoso", "irritado", "irritacao",
    "medo", "medos", "amedrontado", "assustado", "assustador", "aterrorizado",
    "ruim", "horrivel", "pessimo", "horror", "terror", "terrivel",
    "odio", "odeio", "odiar", "detesto", "detestar",
    "mau", "ma", "maus", "mas", "feio", "feia",
    "perigo", "perigoso", "perigosa", "morrer", "morte", "morto", "matar",
    "morta", "sangue", "sangrento", "dor", "dolorido", "sofrer", "sofrimento",
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


def lex_predict(sentences):
    preds = []
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        pos = sum(1 for t in toks if t in POS_WORDS)
        neg = sum(1 for t in toks if t in NEG_WORDS)
        preds.append(0 if neg > pos else (2 if pos > neg else 1))
    return np.array(preds)


print("\n>>> Baseline 1: Léxico simples PT-BR")
r_lex = report("Lexico simples (PT-BR)", y_test, lex_predict(X_test))
with open(ROOT / "baseline_lexico.json", "w", encoding="utf-8") as f:
    json.dump(r_lex, f, indent=2, ensure_ascii=False)
print("Salvo: baseline_lexico.json")


# ---- Baseline 2: SentiLex-PT ----
print("\n>>> Baseline 2: SentiLex-PT (flex, POL:N0)")
pol_re = re.compile(r"POL:N0=(-?1|0)")
votes: dict = defaultdict(list)
with open(SENTILEXPT, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        c = line.find(",")
        if c == -1:
            continue
        word = line[:c].lower()
        m = pol_re.search(line)
        if not m:
            continue
        pol = int(m.group(1))
        if pol != 0:
            votes[word].append(pol)

pol_dict = {}
for word, pols in votes.items():
    p = pols.count(1)
    n = pols.count(-1)
    if p > n:
        pol_dict[word] = 1
    elif n > p:
        pol_dict[word] = -1

pos_count = sum(1 for v in pol_dict.values() if v > 0)
neg_count = sum(1 for v in pol_dict.values() if v < 0)
print(f"  SentiLex: {len(pol_dict)} formas únicas ({pos_count} pos, {neg_count} neg)")


def senti_predict(sentences):
    preds = []
    for s in sentences:
        toks = re.findall(r"\w+", s.lower())
        pos = sum(1 for t in toks if pol_dict.get(t, 0) > 0)
        neg = sum(1 for t in toks if pol_dict.get(t, 0) < 0)
        preds.append(0 if neg > pos else (2 if pos > neg else 1))
    return np.array(preds)


r_senti = report("SentiLex-PT (flex, POL:N0)", y_test, senti_predict(X_test))
with open(ROOT / "baseline_sentilexpt.json", "w", encoding="utf-8") as f:
    json.dump(r_senti, f, indent=2, ensure_ascii=False)
print("Salvo: baseline_sentilexpt.json")

print("\nFeito.")
