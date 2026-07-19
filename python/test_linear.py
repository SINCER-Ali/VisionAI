# Auteur : Valentin BROUC
from bindings import ModeleLineaire                 # importe notre wrapper ctypes

X = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],            # jeu lineairement separable
     [3.0, 3.0], [2.0, 2.5], [4.0, 1.0]]
Y = [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]               # +1 si le point est "en haut a droite"

modele = ModeleLineaire(2)                           # cree le modele (2 entrees)
modele.fit(X, Y, lr=0.0001, epochs=200)                 # entraine

bien = sum(1 for x, y in zip(X, Y) if modele.predict(x) == y)  # compte les bonnes reponses
print(f"{bien}/{len(X)} exemples bien classes")      # doit afficher 6/6