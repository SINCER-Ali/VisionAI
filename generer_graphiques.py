"""
Génération des graphiques de démonstration pour VisionAI.

Produit, à partir des cas de test synthétiques (pas besoin du dataset d'images),
les figures attendues par le cahier des charges :
  - frontières de décision des 4 modèles (linéaire / MLP / RBF / SVM)
  - courbe de convergence (perte et exactitude au fil des époques)
  - sous-apprentissage vs sur-apprentissage
  - comparaison des exactitudes

Prérequis :
    pip install maturin numpy matplotlib scikit-learn
    cd python_binding && maturin develop --release && cd ..
Lancement :
    python generer_graphiques.py
Les images sont enregistrées dans le dossier graphiques/.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import vision_ai

OUT = "graphiques"
os.makedirs(OUT, exist_ok=True)
np.random.seed(42)


def one_hot(labels, n=2):
    out = []
    for l in labels:
        v = [0.0] * n
        v[int(l)] = 1.0
        out.append(v)
    return out


def accuracy(predict_fn, X, y):
    bons = 0
    for xi, yi in zip(X, y):
        p = predict_fn(xi.tolist())
        if (p.index(max(p)) if isinstance(p, list) else int(p[0] > 0.5)) == int(yi):
            bons += 1
    return bons / len(y)


# ───────────────────────── Données : deux lunes (non linéaires) ─────────────
X, y = make_moons(n_samples=400, noise=0.20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
inp_tr = X_tr.tolist()
tgt_tr = one_hot(y_tr)


# ───────────────────────── Figure 1 : frontières de décision ────────────────
def frontiere(ax, predict_fn, titre, lineaire=False):
    h = 0.03
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = []
    for a, b in zip(xx.ravel(), yy.ravel()):
        p = predict_fn([float(a), float(b)])
        Z.append(int(p[0] > 0.5) if lineaire else p.index(max(p)))
    Z = np.array(Z).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolors="k", s=18)
    ax.set_title(titre)
    ax.set_xticks([]); ax.set_yticks([])


fig, axes = plt.subplots(2, 2, figsize=(11, 9))

# Modèle linéaire (régression seuillée à 0.5)
lin = vision_ai.LinearRegression(2)
lin.train(inp_tr, [[float(v)] for v in y_tr], 2000, 0.05)  # (x, y, epochs, lr)
acc_lin = accuracy(lin.predict, X_te, y_te)
frontiere(axes[0, 0], lin.predict, f"Modèle linéaire — acc={acc_lin*100:.0f}%", lineaire=True)

# MLP
mlp = vision_ai.PyMLP([2, 16, 2], "sigmoid")
mlp.train(inp_tr, tgt_tr, 0.5, 3000)  # (inputs, targets, lr, epochs)
acc_mlp = accuracy(mlp.predict, X_te, y_te)
frontiere(axes[0, 1], mlp.predict, f"MLP [2,16,2] — acc={acc_mlp*100:.0f}%")

# RBF
rbf = vision_ai.PyRBF(2, 2, 20, 1.0)
rbf.train(inp_tr, tgt_tr, 0.1, 300)  # (inputs, targets, lr, epochs)
acc_rbf = accuracy(rbf.predict, X_te, y_te)
frontiere(axes[1, 0], rbf.predict, f"RBF (20 centres) — acc={acc_rbf*100:.0f}%")

# SVM noyau RBF
svm = vision_ai.PySVM(c=1.0, kernel="rbf", gamma=2.0)
svm.train(inp_tr, tgt_tr, 0.0, 200)
acc_svm = accuracy(svm.predict, X_te, y_te)
frontiere(axes[1, 1], svm.predict, f"SVM noyau RBF — acc={acc_svm*100:.0f}%")

fig.suptitle("Frontières de décision sur des données non linéaires (deux lunes)", fontsize=14)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "1_frontieres_decision.png"), dpi=130)
print("OK -> 1_frontieres_decision.png")


# ───────────────────────── Figure 2 : courbe de convergence (MLP) ───────────
def perte_ce(predict_fn, X, Y):
    s = 0.0
    for xi, ti in zip(X, Y):
        p = predict_fn(xi.tolist())
        c = ti.index(1.0)
        s += -np.log(max(p[c], 1e-12))
    return s / len(X)


mlp2 = vision_ai.PyMLP([2, 16, 2], "sigmoid")
pas, n_pas = 100, 30
pertes, accs, epochs_axis = [], [], []
for k in range(n_pas):
    mlp2.train(inp_tr, tgt_tr, 0.5, pas)  # continue l'entraînement
    epochs_axis.append((k + 1) * pas)
    pertes.append(perte_ce(mlp2.predict, X_tr, tgt_tr))
    accs.append(accuracy(mlp2.predict, X_te, y_te) * 100)

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(epochs_axis, pertes, "o-", color="crimson", label="Perte (entropie croisée)")
ax1.set_xlabel("Époques"); ax1.set_ylabel("Perte", color="crimson")
ax1.tick_params(axis="y", labelcolor="crimson"); ax1.grid(True, alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(epochs_axis, accs, "s-", color="seagreen", label="Exactitude (test)")
ax2.set_ylabel("Exactitude test (%)", color="seagreen")
ax2.tick_params(axis="y", labelcolor="seagreen"); ax2.set_ylim(0, 100)
plt.title("Convergence du MLP : perte et exactitude au fil des époques")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "2_convergence_mlp.png"), dpi=130)
print("OK -> 2_convergence_mlp.png")


# ───────────────────────── Figure 3 : sous- vs sur-apprentissage ────────────
archs = [("Trop simple [2,2,2]", [2, 2, 2]),
         ("Adapté [2,16,2]", [2, 16, 2]),
         ("Trop complexe [2,128,128,2]", [2, 128, 128, 2])]
acc_train, acc_test = [], []
for _, arch in archs:
    m = vision_ai.PyMLP(arch, "sigmoid")
    m.train(inp_tr, tgt_tr, 0.3, 4000)
    acc_train.append(accuracy(m.predict, X_tr, y_tr) * 100)
    acc_test.append(accuracy(m.predict, X_te, y_te) * 100)

x = np.arange(len(archs)); w = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, acc_train, w, label="Entraînement", color="steelblue")
ax.bar(x + w/2, acc_test, w, label="Test", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels([a[0] for a in archs])
ax.set_ylabel("Exactitude (%)"); ax.set_ylim(0, 105)
ax.set_title("Sous-apprentissage vs sur-apprentissage (écart train/test)")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "3_sur_sous_apprentissage.png"), dpi=130)
print("OK -> 3_sur_sous_apprentissage.png")


# ───────────────────────── Figure 4 : comparaison des modèles ───────────────
noms = ["Linéaire", "MLP", "RBF", "SVM (RBF)"]
valeurs = [acc_lin * 100, acc_mlp * 100, acc_rbf * 100, acc_svm * 100]
fig, ax = plt.subplots(figsize=(8, 5))
barres = ax.bar(noms, valeurs, color=["#9aa0a6", "#1f77b4", "#9467bd", "#d62728"])
ax.set_ylabel("Exactitude test (%)"); ax.set_ylim(0, 105)
ax.set_title("Comparaison des modèles sur les deux lunes")
for b, v in zip(barres, valeurs):
    ax.text(b.get_x() + b.get_width()/2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "4_comparaison_modeles.png"), dpi=130)
print("OK -> 4_comparaison_modeles.png")

print("\nTermine. Figures dans le dossier:", OUT)
