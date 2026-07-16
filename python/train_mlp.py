# Auteur : Thinina
# Entraine le MLP sur les images et sauvegarde (JSON + binaire).
# Prerequis : preprocessing.py et cargo build --release.

import os
import time
import numpy as np
from bindings import MLP

CLASSES = ["aucun", "humain", "animal"]   # aucun=0, humain=1, animal=2

# --- Hyperparametres a manipuler ---
CACHEE = 32          # nb de neurones de la couche cachee
STEPS = 50_000       # nb d'iterations d'entrainement
LR = 0.01            # learning rate


def charger(nom):
    p = os.path.join(os.path.dirname(__file__), "..", "datasets", nom)
    return np.load(p)


def one_hot_pm1(y, n_classes):
    """Etiquette 2 -> [-1, -1, +1] : codage +/-1, adapte a tanh."""
    Y = -np.ones((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def precision(model, X, y):
    bons = 0
    for x, vrai in zip(X, y):
        pred = model.predict(x, is_classification=True)
        if np.argmax(pred) == vrai:                       # classe = plus grande sortie
            bons += 1
    return 100.0 * bons / len(X)


if __name__ == "__main__":
    X_train, y_train = charger("X_train.npy"), charger("y_train.npy")
    X_test, y_test = charger("X_test.npy"), charger("y_test.npy")
    n_entrees = X_train.shape[1]          # 12288 (= 64*64*3)
    n_classes = len(CLASSES)

    print(f"Train : {len(X_train)} images | Test : {len(X_test)} images | {n_entrees} entrees")
    arch = [n_entrees, CACHEE, n_classes]
    print(f"Modele MLP : architecture {arch}  (steps={STEPS}, lr={LR})\n")

    Y_train = one_hot_pm1(y_train, n_classes)

    model = MLP(arch)
    print("Entrainement en cours...")
    t0 = time.perf_counter()
    model.fit(X_train, Y_train, steps=STEPS, lr=LR, is_classification=True)
    duree = time.perf_counter() - t0
    print(f"Duree entrainement : {duree:.1f} s")

    # 2 formats : JSON lisible, binaire compact
    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    model.save_json(os.path.join(dossier, "mlp_weights.json"))    # lisible
    model.save_binary(os.path.join(dossier, "mlp_weights.bin"))   # compact
    print("Modele sauvegarde dans models/ (JSON + binaire)")

    acc_train = precision(model, X_train, y_train)
    acc_test = precision(model, X_test, y_test)
    print(f"\nPrecision entrainement : {acc_train:.1f}%")
    print(f"Precision test         : {acc_test:.1f}%")

    # precision par classe
    print("\nDetail par classe (sur le test) :")
    for etiquette, classe in enumerate(CLASSES):
        idx = np.where(y_test == etiquette)[0]
        if len(idx) == 0:
            print(f"  {classe:8s} : aucune image de test"); continue
        acc = precision(model, X_test[idx], y_test[idx])
        print(f"  {classe:8s} : {acc:5.1f}%  ({len(idx)} images)")

    # Score d'un modele qui repond toujours la classe la plus frequente (baseline a battre)
    vals, cnt = np.unique(y_test, return_counts=True)
    etat = "equilibre" if cnt.max() == cnt.min() else "desequilibre"
    print(f"\nBaseline (repond toujours la classe la plus frequente) : "
          f"{100 * cnt.max() / len(y_test):.1f}%"
          f"  [test {etat} : {'/'.join(str(c) for c in cnt)}]")
