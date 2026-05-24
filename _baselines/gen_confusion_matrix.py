"""
Gera a matriz de confusao do ENSEMBLE (fig03), no mesmo estilo da figura atual.

Estilo replicado de latex/fig03-matriz-confusao.png:
  - heatmap Blues, sem titulo
  - cada celula: contagem (negrito) + porcentagem por linha entre parenteses
  - eixos "Predito" (x) e "Verdadeiro" (y) em negrito
  - barra de cores rotulada "Quantidade"

Matriz do Ensemble A (holdout, 550 amostras) — de ensemble_results.json:
  [[155,  15,   4],
   [ 10, 169,  21],
   [  1,  61, 114]]

Saida: latex/fig03-matriz-confusao.png
  (o arquivo anterior e copiado para fig03-matriz-confusao_modelo-antigo.png)

Execucao:
  cd <projeto>/src
  python -m _baselines.gen_confusion_matrix
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent          # src/_baselines/
PROJECT    = SCRIPT_DIR.parent.parent                 # artigo_03/
LATEX_DIR  = PROJECT / 'latex'
OUT        = LATEX_DIR / 'fig03-matriz-confusao.png'
BACKUP     = LATEX_DIR / 'fig03-matriz-confusao_modelo-antigo.png'

CM = np.array([[155,  15,   4],
               [ 10, 169,  21],
               [  1,  61, 114]])
LABELS = ['Negativo', 'Neutro', 'Positivo']


def main() -> None:
    # anotacao: contagem + porcentagem por linha (recall)
    annot = np.empty_like(CM, dtype=object)
    for i in range(CM.shape[0]):
        row_total = CM[i].sum()
        for j in range(CM.shape[1]):
            pct = 100.0 * CM[i, j] / row_total
            annot[i, j] = f"{CM[i, j]}\n({pct:.1f}%)"

    plt.figure(figsize=(9, 7))
    ax = sns.heatmap(
        CM, annot=annot, fmt='', cmap='Blues',
        xticklabels=LABELS, yticklabels=LABELS,
        linewidths=0.5, linecolor='lightgray',
        annot_kws={'fontsize': 16, 'fontweight': 'bold'},
        cbar_kws={'label': 'Quantidade'},
    )
    ax.set_xlabel('Predito', fontsize=15, fontweight='bold', labelpad=10)
    ax.set_ylabel('Verdadeiro', fontsize=15, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', labelsize=13)
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(13)

    plt.tight_layout()

    if OUT.exists():
        shutil.copy2(OUT, BACKUP)
        print(f"Backup do arquivo anterior -> {BACKUP.name}")

    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusao do ensemble salva -> {OUT}")
    print(f"  acuracia (diagonal/total) = {np.trace(CM)}/{CM.sum()} "
          f"= {100*np.trace(CM)/CM.sum():.2f}%")


if __name__ == '__main__':
    main()
