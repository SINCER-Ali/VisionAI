"""
train_svm.py -- entraine le SVM (un-contre-tous) sur les vraies images, puis
SAUVEGARDE le modele pre-entraine (JSON + binaire) pour l'API.

Le SVM est binaire (+1/-1). Pour les 3 classes, on entraine 1 SVM par classe
(un-contre-tous).

/!\ Le SVM a noyau GARDE tous les exemples d'entrainement -> sur des images
(12288 valeurs) le modele devient tres lourd et lent. On SOUS-ECHANTILLONNE
donc l'entrainement (MAX_EXEMPLES) et on utilise le noyau LINEAIRE (gamma=0),
plus leger. (Limite a discuter dans le rapport.)

Lancer : ../.venv/Scripts/python.exe train_svm.py
"""

import os
import numpy as np
from bindings import SVM, UnContreTous

CLASSES = ["aucun", "humain", "animal"]
LR = 0.001
EPOCHS = 300
GAMMA = 0.001        # noyau RBF (borne [0,1] -> stable, pas de divergence)
MAX_EXEMPLES = 60    # on limite (le SVM stocke tous les exemples)


def charger(nom):
    return np.load(os.path.join(os.path.dirname(__file__), "..", "datasets", nom))


if __name__ == "__main__":
    X, y = charger("X_train.npy"), charger("y_train.npy")
    Xte, yte = charger("X_test.npy"), charger("y_test.npy")

    # sous-echantillonnage (le SVM garde tous les exemples)
    if len(X) > MAX_EXEMPLES:
        idx = np.random.RandomState(0).permutation(len(X))[:MAX_EXEMPLES]
        X, y = X[idx], y[idx]
    print(f"Entrainement SVM (un-contre-tous) sur {len(X)} images "
          f"({X.shape[1]} entrees, noyau {'RBF' if GAMMA > 0 else 'lineaire'})...")

    uct = UnContreTous.entrainer(SVM, X.tolist(), y.tolist(), len(CLASSES),
                                 lr=LR, epochs=EPOCHS, gamma=GAMMA)

    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    uct.save_json(os.path.join(dossier, "svm_weights.json"))
    print("Modele sauvegarde dans models/svm_weights.json")

    bons = sum(int(np.argmax(uct.predict(x)) == int(yv)) for x, yv in zip(Xte.tolist(), yte))
    print(f"Precision test : {100 * bons / len(Xte):.1f}%")
    vals, cnt = np.unique(yte, return_counts=True)
    print(f"Baseline (classe majoritaire) : {100 * cnt.max() / len(yte):.1f}%")
