"""
train_linear.py -- entraine le modele LINEAIRE (un-contre-tous) sur les vraies
images, puis SAUVEGARDE le modele pre-entraine (JSON + binaire) pour l'API.

Le modele lineaire est binaire (+1/-1). Pour les 3 classes, on entraine
1 modele par classe (un-contre-tous) = le "Linear Model x3" du cours.

Lancer : ../.venv/Scripts/python.exe train_linear.py
(preprocessing.py lance avant, et cargo build --release)
"""

import os
import time
import numpy as np
from bindings import ModeleLineaire, UnContreTous

CLASSES = ["aucun", "humain", "animal"]
LR = 0.01        # learning rate
EPOCHS = 50      # passages sur les donnees


def charger(nom):
    return np.load(os.path.join(os.path.dirname(__file__), "..", "datasets", nom))


def precision(uct, X, y):
    """% de bonnes classifications : la classe = le sous-modele au plus grand score."""
    bons = sum(int(np.argmax(uct.predict(x)) == int(vrai))
               for x, vrai in zip(X.tolist(), y))
    return 100.0 * bons / len(X)


if __name__ == "__main__":
    X, y = charger("X_train.npy"), charger("y_train.npy")
    Xte, yte = charger("X_test.npy"), charger("y_test.npy")

    print(f"Train : {len(X)} images | Test : {len(Xte)} images | {X.shape[1]} entrees")
    print(f"Modele LINEAIRE : un-contre-tous ({len(CLASSES)} modeles binaires)"
          f"  (lr={LR}, epochs={EPOCHS})\n")

    print("Entrainement en cours...")
    t0 = time.perf_counter()
    uct = UnContreTous.entrainer(ModeleLineaire, X.tolist(), y.tolist(),
                                 len(CLASSES), lr=LR, epochs=EPOCHS)
    print(f"Duree entrainement : {time.perf_counter() - t0:.1f} s")

    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    uct.save_json(os.path.join(dossier, "lineaire_weights.json"))
    print("Modele sauvegarde dans models/lineaire_weights.json")

    print(f"\nPrecision entrainement : {precision(uct, X, y):.1f}%")
    print(f"Precision test         : {precision(uct, Xte, yte):.1f}%")

    # Precision par classe (pour voir si le modele s'effondre sur une seule classe)
    print("\nDetail par classe (sur le test) :")
    for etiquette, classe in enumerate(CLASSES):
        idx = np.where(yte == etiquette)[0]
        if len(idx) == 0:
            print(f"  {classe:8s} : aucune image de test"); continue
        print(f"  {classe:8s} : {precision(uct, Xte[idx], yte[idx]):5.1f}%  ({len(idx)} images)")

    # Score d'un modele qui repond toujours la classe la plus frequente (baseline a battre)
    vals, cnt = np.unique(yte, return_counts=True)
    etat = "equilibre" if cnt.max() == cnt.min() else "desequilibre"
    print(f"\nBaseline (repond toujours la classe la plus frequente) : "
          f"{100 * cnt.max() / len(yte):.1f}%"
          f"  [test {etat} : {'/'.join(str(c) for c in cnt)}]")
