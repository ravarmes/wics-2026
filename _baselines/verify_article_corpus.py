"""
Verificacao: e possivel reproduzir o 77,27% do artigo (holdout_oficial.json)?

Contexto:
  holdout_oficial.json registra accuracy=0.7727 / f1_macro=0.7749 para o modelo
  AS_sentiment_cv_fold_1_smote_20260515_193838. Re-avaliar esse MESMO modelo
  (run_all_02.py) deu 74,91%. Como os pesos sao fixos, algo na AVALIACAO mudou.

  Ja confirmado: os dois arquivos de corpus de 2750 registros
  (datasets/corpus.csv e data/corpus.csv) produzem holdout identico -> sao o
  mesmo corpus; a diferenca de md5 e so formatacao.

Hipotese testada aqui:
  O modelo foi TREINADO com pre-processamento (train_model() sempre aplica
  preprocess_for_sentiment = limpeza + marcacao de negacoes NEG_). Mas
  evaluate_sentiment.py e run_all_02.py avaliam em TEXTO CRU. Esse descasamento
  treino/avaliacao pode explicar a queda de 77,27% -> 74,91%.

Metodo:
  Avalia o modelo de referencia no holdout do corpus atual em 3 variantes:
    A. texto cru                       (= run_all_02.py, da 74,91%)
    B. limpeza apenas
    C. limpeza + marcacao de negacoes  (= preprocess_for_sentiment completo)
  Funcoes de pre-processamento copiadas verbatim de bertimbau_sentiment.py.

  Se alguma variante reproduzir ~77,27%, o numero do artigo E reproduzivel e a
  "drift" era, na verdade, um bug de avaliacao — nao perda de corpus.

Execucao:
  cd <projeto>/src
  python -m _baselines.verify_article_corpus
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent      # src/_baselines/
SRC_DIR    = SCRIPT_DIR.parent                    # src/
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app.nlp.config import PATHS, get_task_config

REF_MODEL = (SRC_DIR / "app" / "nlp" / "models" / "trained"
             / "AS_sentiment_cv_fold_1_smote_20260515_193838")
CORPUS    = SRC_DIR / "app" / "nlp" / "datasets" / "corpus.csv"

# Alvo registrado em holdout_oficial.json (numero do artigo)
TARGET = {'accuracy': 0.7727272727272727, 'f1_macro': 0.7748822217541185}
TARGET_CM = [[157, 14, 3], [9, 160, 31], [2, 66, 108]]


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
    print("=" * 80)
    print("VERIFICACAO: e possivel reproduzir o 77,27% do artigo?")
    print(f"Modelo de referencia : {REF_MODEL.name}")
    print(f"Alvo (holdout_oficial.json): acc={TARGET['accuracy']*100:.2f}%  "
          f"f1_macro={TARGET['f1_macro']*100:.2f}%")
    print(f"Matriz de confusao alvo    : {TARGET_CM}")
    print("=" * 80)

    if not REF_MODEL.exists():
        print(f"ERRO: modelo de referencia nao encontrado em {REF_MODEL}")
        return

    PATHS['corpus_file'] = str(CORPUS)

    from transformers import BertForSequenceClassification, BertTokenizer

    from app.nlp.datasets.prepare_data_sentiment import get_data_for_cv_and_test
    from app.nlp.evaluation.model_evaluator import ModelEvaluator

    _, (test_texts, test_labels) = get_data_for_cv_and_test()
    test_texts = list(test_texts)

    model     = BertForSequenceClassification.from_pretrained(str(REF_MODEL))
    tokenizer = BertTokenizer.from_pretrained(str(REF_MODEL))
    evaluator = ModelEvaluator(
        task_name='AS',
        class_names=get_task_config('AS')['classes'],
        output_dir=tempfile.mkdtemp(prefix='verify_'),
    )

    variants = {
        'A. texto cru (= run_all_02.py)':
            test_texts,
        'B. limpeza apenas':
            [_clean_text(t) for t in test_texts],
        'C. limpeza + negacoes (preprocess_for_sentiment completo)':
            [_handle_negations(_clean_text(t)) for t in test_texts],
    }

    for name, texts in variants.items():
        res = evaluator.evaluate_model(
            model=model, tokenizer=tokenizer,
            test_texts=texts, test_labels=test_labels,
            max_length=128, batch_size=32,
        )
        acc = res['accuracy']
        f1m = res['f1_macro']
        d_acc = (acc - TARGET['accuracy']) * 100
        d_f1  = (f1m - TARGET['f1_macro']) * 100
        match = abs(d_acc) < 0.10 and abs(d_f1) < 0.10
        f1pc = [round(x, 4) for x in res['f1_per_class']]

        print(f"\n>>> Variante {name}")
        print(f"    accuracy           : {acc*100:.2f}%   "
              f"(delta vs artigo: {d_acc:+.2f} pp)")
        print(f"    f1_macro           : {f1m*100:.2f}%   "
              f"(delta vs artigo: {d_f1:+.2f} pp)")
        print(f"    f1 por classe      : Neg={f1pc[0]}  Neu={f1pc[1]}  Pos={f1pc[2]}")
        print(f"    matriz de confusao : {res['confusion_matrix']}")
        print(f"    >>> {'BATE com o artigo' if match else 'NAO bate'}")

    print("\n" + "=" * 80)
    print("Se alguma variante BATE -> 77,27% e reproduzivel (era bug de avaliacao).")
    print("Se nenhuma bate        -> o holdout do artigo nao e reconstruivel.")
    print("=" * 80)


if __name__ == '__main__':
    main()
