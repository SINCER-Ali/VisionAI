"""
Smoke test du binding vision_ai : teste PyMLP, PyRBF et PySVM sur des
donnees SYNTHETIQUES (pas besoin du vrai dataset), et sauvegarde 3 modeles
dans models/ pour pouvoir tester le serveur + le client de bout en bout.

Usage :
    1. construire le binding :  cd python_binding && maturin develop --release
    2. revenir a la racine    :  cd ..
    3. lancer                 :  python test_binding_smoke.py
"""
import os
import numpy as np
import vision_ai

np.random.seed(0)
DIM = 12288          # 64 x 64 x 3, comme une vraie image
N_PER_CLASS = 30
CLASSES = ['aucun', 'humain', 'animal']

# --- Donnees synthetiques apprenables : chaque classe "allume" un tiers des pixels ---
X, y = [], []
for cls in range(3):
    for _ in range(N_PER_CLASS):
        v = np.random.rand(DIM) * 0.1                 # bruit faible
        start = cls * (DIM // 3)
        v[start:start + DIM // 3] += 0.8              # motif distinctif
        X.append(v.tolist())
        y.append(cls)

targets = [[1.0 if i == cls else 0.0 for i in range(3)] for cls in y]

def accuracy(predict_fn):
    preds = [predict_fn(x) for x in X]
    y_pred = [p.index(max(p)) for p in preds]
    return sum(p == t for p, t in zip(y_pred, y)) / len(y) * 100

os.makedirs('models', exist_ok=True)

# --- MLP ---
print('--- MLP ---')
mlp = vision_ai.PyMLP([DIM, 64, 3])
mlp.train(X, targets, 0.01, 30)
print(f'  accuracy (synthetique) : {accuracy(mlp.predict):.0f}%')
mlp.save_json('models/mlp_weights.json')
print('  -> models/mlp_weights.json')

# --- RBF (sigma grand car haute dimension) ---
print('--- RBF ---')
rbf = vision_ai.PyRBF(DIM, 3, n_centers=20, sigma=20.0)
rbf.init_centers_random(X)           # no-op (init faite dans train), gardé pour compat
rbf.train(X, targets, 0.01, 50, False)
print(f'  accuracy (synthetique) : {accuracy(rbf.predict):.0f}%')
rbf.save_json('models/rbf_weights.json')
print('  -> models/rbf_weights.json')

# --- SVM (lineaire) ---
print('--- SVM ---')
svm = vision_ai.PySVM(c=1.0, kernel='linear')
svm.train(X, targets, 0.01, 50)
print(f'  accuracy (synthetique) : {accuracy(svm.predict):.0f}%')
svm.save_json('models/svm_weights.json')
print('  -> models/svm_weights.json')

print('\nOK ! Les 3 modeles sont sauvegardes dans models/.')
print('Lance maintenant le serveur (cargo run -p api_server) puis ouvre client/index.html.')
