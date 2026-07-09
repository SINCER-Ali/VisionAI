"""
preprocessing.py -- transforme les images en tableaux .npy pour le MLP.

Chaine : datasets/train/{aucun,humain,animal}/*.jpg
      -> redimension 64x64x3, normalisation [0,1], aplatissement (12288 valeurs)
      -> decoupage train / test
      -> datasets/X_train.npy, y_train.npy, X_test.npy, y_test.npy

Lancer :  ../.venv/Scripts/python.exe preprocessing.py
A relancer a chaque fois que tu ajoutes des photos.
"""

import os
import numpy as np
from PIL import Image

# --- Reglages ---
DOSSIER = os.path.join(os.path.dirname(__file__), "..", "datasets", "train")
CLASSES = ["aucun", "humain", "animal"]   # l'indice = l'etiquette : aucun=0, humain=1, animal=2
TAILLE = 64                               # images redimensionnees en TAILLE x TAILLE
PART_TEST = 0.2                           # 20% des images pour le test
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
np.random.seed(42)                        # decoupage reproductible


def image_to_vector(chemin, taille=TAILLE):
    """UNE image -> un vecteur de taille*taille*3 valeurs dans [0,1].
    Utilisee ici (dataset) ET par l'API (image envoyee par l'utilisateur)."""
    img = Image.open(chemin).convert("RGB").resize((taille, taille))  # RGB + redimension
    return np.asarray(img, dtype=np.float64).ravel() / 255.0          # aplati + normalise


def charger_images():
    """Parcourt les dossiers et renvoie X (images aplaties) et y (etiquettes)."""
    X, y = [], []
    for etiquette, classe in enumerate(CLASSES):
        dossier_classe = os.path.join(DOSSIER, classe)
        fichiers = [f for f in os.listdir(dossier_classe) if f.lower().endswith(EXTS)]
        print(f"  {classe:8s} (label {etiquette}) : {len(fichiers)} images")
        for nom in fichiers:
            chemin = os.path.join(dossier_classe, nom)
            try:
                X.append(image_to_vector(chemin))  # meme conversion que l'API
            except Exception as e:
                print(f"    /!\\ image ignoree ({nom}) : {e}")
                continue
            y.append(etiquette)
    return np.array(X), np.array(y)


def decouper(X, y):
    """Decoupe en train/test en gardant la proportion de chaque classe (stratifie)."""
    Xtr, ytr, Xte, yte = [], [], [], []
    for etiquette in range(len(CLASSES)):
        idx = np.where(y == etiquette)[0]
        np.random.shuffle(idx)
        n_test = int(len(idx) * PART_TEST)
        idx_test, idx_train = idx[:n_test], idx[n_test:]
        Xte.append(X[idx_test]); yte.append(y[idx_test])
        Xtr.append(X[idx_train]); ytr.append(y[idx_train])
    return (np.concatenate(Xtr), np.concatenate(ytr),
            np.concatenate(Xte), np.concatenate(yte))


if __name__ == "__main__":
    print("Chargement des images...")
    X, y = charger_images()
    print(f"Total : {len(X)} images de {X.shape[1] if len(X) else 0} valeurs chacune")

    X_train, y_train, X_test, y_test = decouper(X, y)

    sortie = os.path.join(os.path.dirname(__file__), "..", "datasets")
    np.save(os.path.join(sortie, "X_train.npy"), X_train)
    np.save(os.path.join(sortie, "y_train.npy"), y_train)
    np.save(os.path.join(sortie, "X_test.npy"), X_test)
    np.save(os.path.join(sortie, "y_test.npy"), y_test)

    print(f"\nEnregistre : {len(X_train)} images d'entrainement, {len(X_test)} de test.")
    print("Fichiers .npy regeneres dans datasets/.")
