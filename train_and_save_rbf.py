import numpy as np
import vision_ai
import os

# Entraine un RBF sur le vrai dataset et sauvegarde les poids pour l'API.
# IMPORTANT : on entraine sur la dimension PLEINE (12288) pour rester
# compatible avec les images envoyees par l'API (64x64x3 = 12288).

X_train = np.load('datasets/X_train.npy')
y_train = np.load('datasets/y_train.npy')

inputs = [X_train[i].tolist() for i in range(len(X_train))]
targets = []
for label in y_train:
    t = [0.0, 0.0, 0.0]
    t[int(label)] = 1.0
    targets.append(t)

print('Entrainement RBF...')
# sigma grand car haute dimension (gamma = 1/(2*sigma^2) reste petit)
model = vision_ai.PyRBF(X_train.shape[1], 3, n_centers=30, sigma=20.0)
model.init_centers_random(inputs)
model.train(inputs, targets, 0.01, 50, False)
print('Termine !')

X_test = np.load('datasets/X_test.npy')
y_test = np.load('datasets/y_test.npy')

inputs_test = [X_test[i].tolist() for i in range(len(X_test))]
predictions = [model.predict(x) for x in inputs_test]
y_pred = [pred.index(max(pred)) for pred in predictions]
correct = sum(1 for p, t in zip(y_pred, y_test) if p == int(t))
accuracy = correct / len(y_test) * 100
print(f'Accuracy : {accuracy:.1f}%')

os.makedirs('models', exist_ok=True)

model.save_json('models/rbf_weights.json')
print('Poids sauvegardes dans models/rbf_weights.json !')
