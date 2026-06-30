"""
Entraine un MLP sur le dataset et sauvegarde la matrice de confusion en image.
Resultat : graphiques/matrice_confusion.png (affichee dans index.html)

Prerequis : binding construit + fichiers .npy generes.
Lancement  : python matrice_confusion.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import vision_ai

DATASET_DIR = "datasets"
CLASSES = ["aucun", "humain", "animal"]

X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
X_test = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
y_test = np.load(os.path.join(DATASET_DIR, "y_test.npy"))

INPUT_SIZE = X_train.shape[1]
inputs_train = X_train.tolist()
inputs_test = X_test.tolist()
targets_train = [[1.0 if int(l) == k else 0.0 for k in range(3)] for l in y_train]

print("Entrainement du MLP...")
modele = vision_ai.PyMLP([INPUT_SIZE, 64, 3], "sigmoid")
modele.train(inputs_train, targets_train, 0.01, 400)

y_pred = [modele.predict(x) for x in inputs_test]
y_pred = [p.index(max(p)) for p in y_pred]

acc = accuracy_score(y_test, y_pred) * 100
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
ax.set_xlabel("Predit"); ax.set_ylabel("Vraie classe")
ax.set_title(f"Matrice de confusion - MLP ({acc:.1f}% sur le test)")
for i in range(3):
    for j in range(3):
        couleur = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=couleur, fontweight="bold")
fig.colorbar(im)
fig.tight_layout()

os.makedirs("graphiques", exist_ok=True)
fig.savefig("graphiques/matrice_confusion.png", dpi=130)
print(f"Exactitude test : {acc:.1f}%")
print("Image enregistree : graphiques/matrice_confusion.png")
