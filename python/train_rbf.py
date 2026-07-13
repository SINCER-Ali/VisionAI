# un rbf par classe (one-vs-rest) sur les images, puis sauvegarde
import os
import json
import numpy as np
from bindings import RBFNetwork

CLASSES = ["aucun", "humain", "animal"]   # aucun=0, humain=1, animal=2

K = 20          # nb de centres (k-means)
GAMMA = 0.01    # largeur des gaussiennes
ITERS = 15      # iterations du k-means


def charger(nom):
    p = os.path.join(os.path.dirname(__file__), "..", "datasets", nom)
    return np.load(p)


def entraine_un(X, y, classe):
    # cible +1 pour la classe visee, -1 pour les autres (one-vs-rest)
    cible = np.where(y == classe, 1.0, -1.0)
    m = RBFNetwork(k=K, gamma=GAMMA)
    m.train(X.tolist(), cible.tolist(), k=K, iterations=ITERS)
    return m


def predit(modeles, x):
    # classe = celle dont le RBF donne la plus grande sortie
    scores = [m.predict(list(x)) for m in modeles]
    return int(np.argmax(scores))


if __name__ == "__main__":
    X_train, y_train = charger("X_train.npy"), charger("y_train.npy")
    X_test, y_test = charger("X_test.npy"), charger("y_test.npy")
    print(f"Train {len(X_train)} | Test {len(X_test)} | {X_train.shape[1]} entrees")

    modeles = [entraine_un(X_train, y_train, c) for c in range(len(CLASSES))]

    bons = sum(predit(modeles, x) == vrai for x, vrai in zip(X_test, y_test))
    print(f"Precision test : {100.0 * bons / len(X_test):.1f}%")

    # sauvegarde des 3 modeles (json lisible + binaire par modele)
    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(dossier, exist_ok=True)
    etats = [m.get_state() for m in modeles]
    with open(os.path.join(dossier, "rbf_weights.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": CLASSES, "modeles": etats}, f)
    for i, m in enumerate(modeles):
        m.save_binary(os.path.join(dossier, f"rbf_{i}.bin"))
    print("Modeles sauvegardes dans models/ (rbf_weights.json + rbf_0/1/2.bin)")
