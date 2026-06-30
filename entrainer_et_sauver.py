"""
Entraine un MLP sur le dataset et le sauvegarde pour le serveur.
Le serveur chargera ce fichier au demarrage : models/mlp_weights.json

Prerequis : binding construit + fichiers .npy generes (en 64x64).
Lancement  : python entrainer_et_sauver.py
"""

import os
import numpy as np

import vision_ai

DATASET_DIR = "datasets"

X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
INPUT_SIZE = X_train.shape[1]

if INPUT_SIZE != 12288:
    print(f"ATTENTION : taille {INPUT_SIZE} (le serveur attend 12288 = 64x64).")
    print("Remets IMG_SIZE = 64 dans preprocess_dataset.py et relance le preprocessing.")

inputs_train = X_train.tolist()
targets_train = [[1.0 if int(l) == k else 0.0 for k in range(3)] for l in y_train]

print("Entrainement du MLP...")
modele = vision_ai.PyMLP([INPUT_SIZE, 64, 3], "sigmoid")
modele.train(inputs_train, targets_train, 0.01, 400)

os.makedirs("models", exist_ok=True)
modele.save_json("models/mlp_weights.json")
print("Modele sauvegarde dans models/mlp_weights.json")
print("Tu peux maintenant lancer le serveur : cargo run --release -p api_server")
