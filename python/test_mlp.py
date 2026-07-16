# Auteur : Thinina
# test du MLP : verification rapide sur les cas cles (separable, XOR, multi-classe, regression)

from bindings import MLP                                          # importe le wrapper MLP

# --- Test 1 : Linear Simple (separable) -> pas besoin de couche cachee ---
X = [[1.0, 1.0], [2.0, 3.0], [3.0, 3.0]]                          # 3 points 2D
Y = [[1.0], [-1.0], [-1.0]]                                       # leurs classes (+1 / -1)
m = MLP([2, 1])                                                   # aucune couche cachee
m.fit(X, Y, steps=50_000, lr=0.01, is_classification=True)        # entraine
bien = sum(1 for x, y in zip(X, Y) if (m.predict(x)[0] >= 0) == (y[0] >= 0))
print(f"Linear Simple : {bien}/{len(X)} bien classes")            # vise 3/3

# --- Test 2 : XOR (non separable) -> il faut une couche cachee ---
# [2,2,1] se coince parfois dans un minimum local selon l'init -> on reessaie.
Xxor = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 1.0]]           # les 4 points du XOR
Yxor = [[1.0], [1.0], [-1.0], [-1.0]]                             # classes du XOR
bienxor = 0
for essai in range(5):                                            # jusqu'a 5 initialisations
    mx = MLP([2, 2, 1])                                           # 2 neurones caches
    mx.fit(Xxor, Yxor, steps=300_000, lr=0.01, is_classification=True)
    bienxor = sum(1 for x, y in zip(Xxor, Yxor) if (mx.predict(x)[0] >= 0) == (y[0] >= 0))
    if bienxor == len(Xxor):                                      # tous bons -> on s'arrete
        break
print(f"XOR : {bienxor}/{len(Xxor)} bien classes  (essai {essai + 1})")   # vise 4/4

# --- Test 3 : multi-classe (le MLP a 3 sorties -> natif, pas besoin d'un-contre-tous) ---
Xm = [[0.0, 0.0], [0.2, 0.1], [2.0, 2.0], [2.1, 1.9], [4.0, 0.0], [3.9, 0.2]]   # 3 groupes
Ym = [[1.0, -1.0, -1.0], [1.0, -1.0, -1.0],                       # classe 0 en one-hot +/-1
      [-1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],                       # classe 1
      [-1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]]                       # classe 2
mm = MLP([2, 4, 3])                                               # 3 neurones de sortie
mm.fit(Xm, Ym, steps=100_000, lr=0.01, is_classification=True)
bienm = sum(1 for x, y in zip(Xm, Ym)
            if mm.predict(x).index(max(mm.predict(x))) == y.index(max(y)))   # argmax
print(f"Multi-classe : {bienm}/{len(Xm)} bien classes")           # vise 6/6

# --- Test 4 : regression (sortie LINEAIRE, pas de tanh sur la derniere couche) ---
Xr = [[1.0], [2.0], [3.0]]                                        # 1 entree
Yr = [[2.0], [4.0], [6.0]]                                        # y = 2x
mr = MLP([1, 1])                                                  # sans couche cachee
mr.fit(Xr, Yr, steps=50_000, lr=0.01, is_classification=False)    # <- False = regression
mse = sum((mr.predict(x, is_classification=False)[0] - y[0]) ** 2 for x, y in zip(Xr, Yr)) / len(Xr)
print(f"Regression y=2x : MSE = {mse:.6f}")                       # vise ~0
