"""
Entraine un modele sur le dataset d'images et l'evalue.
Affiche l'exactitude (paquet test) et la matrice de confusion.

Prerequis :
  - binding construit (cd python_binding && maturin develop --release)
  - fichiers .npy generes (python preprocess_dataset.py)
Lancement :
  python tester_dataset.py
"""

import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

import vision_ai

DATASET_DIR = "datasets"
CLASSES = ["aucun", "humain", "animal"]

X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

INPUT_SIZE = X_train.shape[1]
print(f"Entrainement : {X_train.shape[0]} images | Test : {X_test.shape[0]} images")
print(f"Taille d'une image : {INPUT_SIZE} nombres ({CLASSES})")

inputs_train = X_train.tolist()
inputs_test = X_test.tolist()


def one_hot(labels, n=3):
    out = []
    for l in labels:
        v = [0.0] * n
        v[int(l)] = 1.0
        out.append(v)
    return out


targets_train = one_hot(y_train)

print("\nEntrainement du MLP en cours...")
modele = vision_ai.PyMLP([INPUT_SIZE, 128, 3], "sigmoid")
modele.train(inputs_train, targets_train, 0.01, 200)

y_pred = [modele.predict(x).index(max(modele.predict(x))) for x in inputs_test]

acc = accuracy_score(y_test, y_pred)
print(f"\nExactitude sur le paquet test : {acc * 100:.1f}%")

print("\nMatrice de confusion (lignes = vraie classe, colonnes = predit) :")
cm = confusion_matrix(y_test, y_pred)
entete = "            " + "  ".join(f"{c[:6]:>6}" for c in CLASSES)
print(entete)
for i, ligne in enumerate(cm):
    print(f"{CLASSES[i]:>10}  " + "  ".join(f"{v:>6}" for v in ligne))

print("\nTermine.")
