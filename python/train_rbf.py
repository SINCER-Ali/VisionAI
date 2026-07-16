# un rbf par classe (one-vs-rest) sur les images, puis sauvegarde
import os
import time
import numpy as np
from bindings import RBFNetwork, UnContreTous

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


def precision(modeles, X, y):
    """% de bonnes classifications."""
    bons = sum(predit(modeles, x) == vrai for x, vrai in zip(X, y))
    return 100.0 * bons / len(X)


if __name__ == "__main__":
    X_train, y_train = charger("X_train.npy"), charger("y_train.npy")
    X_test, y_test = charger("X_test.npy"), charger("y_test.npy")

    print(f"Train : {len(X_train)} images | Test : {len(X_test)} images | "
          f"{X_train.shape[1]} entrees")
    print(f"Modele RBF : un-contre-tous ({len(CLASSES)} reseaux)"
          f"  (k={K}, gamma={GAMMA}, iters={ITERS})\n")

    print("Entrainement en cours...")
    t0 = time.perf_counter()
    modeles = [entraine_un(X_train, y_train, c) for c in range(len(CLASSES))]
    print(f"Duree entrainement : {time.perf_counter() - t0:.1f} s")

    # Sauvegarde du modele entraine (2 formats) -> l'API le chargera sans re-entrainer.
    # On passe par UnContreTous : meme mecanique que le lineaire et le SVM, et les
    # 3 reseaux tiennent dans UN SEUL fichier (au lieu d'un fichier par classe).
    dossier = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(dossier, exist_ok=True)
    uct = UnContreTous(modeles, CLASSES)
    uct.save_json(os.path.join(dossier, "rbf_weights.json"))          # lisible
    uct.save_binary(os.path.join(dossier, "rbf_weights.bin"))         # compact
    print("Modele sauvegarde dans models/ (JSON + binaire)")

    print(f"\nPrecision entrainement : {precision(modeles, X_train, y_train):.1f}%")
    print(f"Precision test         : {precision(modeles, X_test, y_test):.1f}%")

    # Precision par classe (pour voir si le modele s'effondre sur une seule classe)
    print("\nDetail par classe (sur le test) :")
    for etiquette, classe in enumerate(CLASSES):
        idx = np.where(y_test == etiquette)[0]
        if len(idx) == 0:
            print(f"  {classe:8s} : aucune image de test"); continue
        print(f"  {classe:8s} : {precision(modeles, X_test[idx], y_test[idx]):5.1f}%"
              f"  ({len(idx)} images)")

    # Score d'un modele qui repond toujours la classe la plus frequente (baseline a battre)
    vals, cnt = np.unique(y_test, return_counts=True)
    etat = "equilibre" if cnt.max() == cnt.min() else "desequilibre"
    print(f"\nBaseline (repond toujours la classe la plus frequente) : "
          f"{100.0 * cnt.max() / len(y_test):.1f}%"
          f"  [test {etat} : {'/'.join(str(c) for c in cnt)}]")
