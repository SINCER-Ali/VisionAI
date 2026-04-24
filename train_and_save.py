import numpy as np
import vision_ai
import json
import os

X_train = np.load('datasets/X_train.npy')
y_train = np.load('datasets/y_train.npy')

inputs = [X_train[i].tolist() for i in range(len(X_train))]
targets = []
for label in y_train:
    t = [0.0, 0.0, 0.0]
    t[int(label)] = 1.0
    targets.append(t)

print('Entrainement...')
model = vision_ai.PyMLP([12288, 64, 3])
model.train(inputs, targets, 0.001, 10)
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

model.save_json('models/mlp_weights.json')
print('Poids sauvegardes dans models/mlp_weights.json !')

weights_data = {'layer_sizes': [12288, 64, 3], 'accuracy': accuracy}
with open('models/mlp_config.json', 'w') as f:
    json.dump(weights_data, f)
print('Config sauvegardee !')