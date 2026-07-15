"""
train_linear.py -- entraine le modele LINEAIRE (un-contre-tous) sur les vraies
images, puis SAUVEGARDE le modele pre-entraine (JSON + binaire) pour l'API.

Le modele lineaire est binaire (+1/-1). Pour les 3 classes, on entraine
1 modele par classe (un-contre-tous) = le "Linear Model x3" du cours.

Lancer : ../.venv/Scripts/python.exe train_linear.py
(preprocessing.py lance avant, et cargo build --release)
"""

import os
import numpy as np
from bindings import ModeleLineaire, UnContreTous

CLASSES = ["aucun", "humain", "animal"]
LR = 0.01        # learning rate
EPOCHS = 50      # passages sur les donnees


def charger(nom):
    return np.load(os.path.join(os.path.dirname(__file__), "..", "datasets", nom))


if __name__ == "__main__":
    X, y = charger("X_train.npy"), charger("y_train.npy")
    Xte, yte = charger("X_test.npy"), charger("y_test.npy")
    print(f"Entrainement LINEAIRE (un-contre-tous) sur {len(X)} images "
          f"({X.shape[1]} entrees)...")

    uct = UnContreTous.entrainer(ModeleLineaire, X.tolist(), y.tolist(),
                                 len(CLASSES), lr=LR, epochs=EPOCHS)

    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    uct.save_json(os.path.join(dossier, "lineaire_weights.json"))
    print("Modele sauvegarde dans models/lineaire_weights.json")

    # precision sur le test
    bons = sum(int(np.argmax(uct.predict(x)) == int(yv)) for x, yv in zip(Xte.tolist(), yte))
    print(f"Precision test : {100 * bons / len(Xte):.1f}%")
    vals, cnt = np.unique(yte, return_counts=True)
    print(f"Baseline (classe majoritaire) : {100 * cnt.max() / len(yte):.1f}%")
