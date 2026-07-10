# test_rbf ali
# test du rbf sur le xor pour verifier que ca marche

import sys
sys.path.append('..')
from bindings import RBFNetwork

# donnees XOR : 4 points, 2 classes
data = [
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
]
targets = [-1.0, 1.0, 1.0, -1.0]  # xor : different = 1, pareil = -1

# on cree et entraine le modele
rbf = RBFNetwork(k=2, gamma=2.0)
rbf.train(data, targets, k=4, iterations=100)

# on teste les predictions
for x, expected in zip(data, targets):
    pred = rbf.predict_class(x)
    print(f"input {x} → predit {pred}, attendu {expected}, {'OK' if pred == expected else 'FAUX'}")