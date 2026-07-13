# Auteur : Valentin BROUC
from bindings import SVM                                   # importe le wrapper SVM

# --- Test 1 : cas lineairement separable (noyau lineaire, gamma=0) ---
X = [[0.0,0.0],[1.0,0.0],[0.0,1.0],[3.0,3.0],[2.0,2.5],[4.0,1.0]]  # 6 points 2D
Y = [-1.0,-1.0,-1.0,1.0,1.0,1.0]                            # leurs classes
m = SVM()                                                   # cree le SVM
m.fit(X, Y, lr=0.001, epochs=2000, gamma=0.0)               # entraine, noyau lineaire
bien = sum(1 for x, y in zip(X, Y) if m.predict(x) == y)    # compte les bonnes reponses
print(f"Lineaire : {bien}/{len(X)} bien classes")           # vise 6/6

# --- Test 2 : XOR (non separable) avec noyau RBF (gamma>0) ---
Xxor = [[0.0,0.0],[1.0,1.0],[1.0,0.0],[0.0,1.0]]            # les 4 points du XOR
Yxor = [-1.0,-1.0,1.0,1.0]                                  # classes du XOR
mx = SVM()                                                  # nouveau SVM
mx.fit(Xxor, Yxor, lr=0.5, epochs=5000, gamma=1.0)          # entraine, noyau RBF (gamma>0)
bienxor = sum(1 for x, y in zip(Xxor, Yxor) if mx.predict(x) == y)
print(f"XOR (RBF) : {bienxor}/{len(Xxor)} bien classes")    # vise 4/4