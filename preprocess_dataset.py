import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

IMG_SIZE = 64
DATASET_DIR = "datasets/train"
OUTPUT_DIR = "datasets"

CLASSES = {"aucun": 0, "humain": 1, "animal": 2}

images = []
labels = []

print("Chargement des images...")

for classe, label in CLASSES.items():
    dossier = os.path.join(DATASET_DIR, classe)
    fichiers = os.listdir(dossier)[:500]
    print(f"  {classe} : {len(fichiers)} images")

    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        try:
            img = Image.open(chemin).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            vecteur = np.array(img).flatten() / 255.0
            images.append(vecteur)
            labels.append(label)
        except:
            pass

X = np.array(images)
y = np.array(labels)

print(f"\nTotal images : {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

print(f"Train : {len(X_train)} images")
print(f"Test  : {len(X_test)} images")
print("Fichiers .npy sauvegardes !")