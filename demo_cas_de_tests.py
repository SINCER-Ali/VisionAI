"""
Démonstration visuelle des cas de test.

Pour chaque cas (XOR, AND, OR...), on entraîne le modèle, on affiche
chaque tentative, puis le détail entrée par entrée : prédit vs attendu.

Prérequis : binding construit (cd python_binding && maturin develop --release)
Lancement  : python demo_cas_de_tests.py
"""

import vision_ai

MAX_TENTATIVES = 5


def one_hot(label, n=2):
    v = [0.0] * n
    v[label] = 1.0
    return v


def classe_predite(proba):
    return proba.index(max(proba))


def run_cas(nom, inputs, labels, fabrique_modele, lr, epochs):
    """Entraîne (jusqu'à 5 tentatives) et affiche le détail."""
    print("\n" + "=" * 55)
    print(f"  CAS DE TEST : {nom}")
    print("=" * 55)

    targets = [one_hot(l) for l in labels]

    for tentative in range(1, MAX_TENTATIVES + 1):
        modele = fabrique_modele()
        modele.train(inputs, targets, lr, epochs)
        predits = [classe_predite(modele.predict(x)) for x in inputs]
        nb_ok = sum(p == l for p, l in zip(predits, labels))
        print(f"  Tentative {tentative}/{MAX_TENTATIVES} : {nb_ok}/{len(labels)} corrects", end="")

        if nb_ok == len(labels):
            print("   -> CONVERGE")
            print("  " + "-" * 45)
            for x, p, l in zip(inputs, predits, labels):
                etat = "OK" if p == l else "KO"
                conf = max(modele.predict(x)) * 100
                print(f"    entree {x} -> predit {p} (attendu {l})  [{etat}, {conf:.0f}%]")
            print(f"  => REUSSI en {tentative} tentative(s)")
            return True
        else:
            print("   -> echec, on retente")

    print(f"  => NON CONVERGE apres {MAX_TENTATIVES} tentatives")
    return False


# ─────────────────────── Les données des cas de test ───────────────────────
XOR = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 0])
AND = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 0, 0, 1])
OR  = ([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], [0, 1, 1, 1])


# ─────────────────────── Les modèles à tester ──────────────────────────────
def mlp():
    return vision_ai.PyMLP([2, 16, 2], "sigmoid")

def rbf():
    return vision_ai.PyRBF(2, 2, 4, 2.0)

def svm_rbf():
    return vision_ai.PySVM(c=5.0, kernel="rbf", gamma=1.0)


print("####################################################")
print("#   DEMONSTRATION DES CAS DE TEST - VisionAI       #")
print("####################################################")

# MLP sur les 3 problèmes logiques
run_cas("XOR  (MLP)", XOR[0], XOR[1], mlp, lr=0.5, epochs=3000)
run_cas("AND  (MLP)", AND[0], AND[1], mlp, lr=0.5, epochs=2000)
run_cas("OR   (MLP)", OR[0],  OR[1],  mlp, lr=0.5, epochs=2000)

# XOR (non linéaire) avec les autres modèles
run_cas("XOR  (RBF)", XOR[0], XOR[1], rbf, lr=0.05, epochs=300)
run_cas("XOR  (SVM noyau RBF)", XOR[0], XOR[1], svm_rbf, lr=0.0, epochs=200)

print("\nTermine.")
