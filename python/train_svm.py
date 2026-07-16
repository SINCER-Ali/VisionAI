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
import time
import numpy as np
from bindings import SVM, UnContreTous

CLASSES = ["aucun", "humain", "animal"]
LR = 0.001
EPOCHS = 300
GAMMA = 0.001        # noyau RBF (borne [0,1] -> stable, pas de divergence)
MAX_EXEMPLES = 60    # on limite (le SVM stocke tous les exemples)


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
    n_dispo = len(X)

    # sous-echantillonnage (le SVM garde tous les exemples -> cout en O(n^2))
    if len(X) > MAX_EXEMPLES:
        idx = np.random.RandomState(0).permutation(len(X))[:MAX_EXEMPLES]
        X, y = X[idx], y[idx]

    print(f"Train : {len(X)} images (sous-echantillonnees sur {n_dispo}) | "
          f"Test : {len(Xte)} images | {X.shape[1]} entrees")
    print(f"Modele SVM : un-contre-tous ({len(CLASSES)} modeles binaires), "
          f"noyau {'RBF' if GAMMA > 0 else 'lineaire'}"
          f"  (lr={LR}, epochs={EPOCHS}, gamma={GAMMA})\n")

    print("Entrainement en cours...")
    t0 = time.perf_counter()
    uct = UnContreTous.entrainer(SVM, X.tolist(), y.tolist(), len(CLASSES),
                                 lr=LR, epochs=EPOCHS, gamma=GAMMA)
    print(f"Duree entrainement : {time.perf_counter() - t0:.1f} s")

    # Sauvegarde du modele entraine (2 formats) -> l'API le chargera sans re-entrainer.
    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    uct.save_json(os.path.join(dossier, "svm_weights.json"))          # lisible
    uct.save_binary(os.path.join(dossier, "svm_weights.bin"))         # compact
    print("Modele sauvegarde dans models/ (JSON + binaire)")

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
