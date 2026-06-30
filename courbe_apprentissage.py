"""
Trace la courbe d'apprentissage du MLP sur le dataset d'images :
perte (entrainement) et exactitude (test) au fil des epoques.

Prerequis : binding construit + fichiers .npy generes.
Lancement  : python courbe_apprentissage.py
Astuce vitesse : mettre IMG_SIZE=32 dans preprocess_dataset.py et regenerer.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import vision_ai

DATASET_DIR = "datasets"

X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

INPUT_SIZE = X_train.shape[1]
inputs_train = X_train.tolist()
inputs_test = X_test.tolist()


def one_hot(labels, n=3):
    return [[1.0 if int(l) == k else 0.0 for k in range(n)] for l in labels]


targets_train = one_hot(y_train)

PAS = 10
NB_PAS = 40
LR = 0.01

os.makedirs("graphiques", exist_ok=True)


def sauver_courbe(epoques, pertes, exactitudes):
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epoques, pertes, "o-", color="crimson")
    ax1.set_xlabel("Epoques")
    ax1.set_ylabel("Perte", color="crimson")
    ax1.tick_params(axis="y", labelcolor="crimson")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epoques, exactitudes, "s-", color="seagreen")
    ax2.set_ylabel("Exactitude test (%)", color="seagreen")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2.set_ylim(0, 100)
    plt.title("Courbe d'apprentissage du MLP sur le dataset")
    fig.tight_layout()
    fig.savefig("graphiques/courbe_apprentissage_dataset.png", dpi=130)
    plt.close(fig)


modele = vision_ai.PyMLP([INPUT_SIZE, 64, 3], "sigmoid")

epoques, pertes, exactitudes = [], [], []
for k in range(NB_PAS):
    modele.train(inputs_train, targets_train, LR, PAS)

    perte = 0.0
    for x, t in zip(inputs_train, targets_train):
        p = modele.predict(x)
        c = t.index(1.0)
        perte += -np.log(max(p[c], 1e-12))
    perte /= len(inputs_train)

    y_pred = [modele.predict(x) for x in inputs_test]
    y_pred = [p.index(max(p)) for p in y_pred]
    acc = accuracy_score(y_test, y_pred) * 100

    epoques.append((k + 1) * PAS)
    pertes.append(perte)
    exactitudes.append(acc)
    print(f"epoque {epoques[-1]:4d}  |  perte {perte:.3f}  |  exactitude test {acc:.1f}%")

    sauver_courbe(epoques, pertes, exactitudes)

print("\nTermine. Image mise a jour a chaque palier : graphiques/courbe_apprentissage_dataset.png")
