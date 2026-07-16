# Pseudo-code du MLP / PMC (Perceptron Multi-Couches) — Auteur : Thinina

> Support de maîtrise. Chaque partie du pseudo-code correspond **ligne pour ligne**
> à `lib_rust/src/mlp.rs`. But : pouvoir réciter et modifier le MLP devant le jury.

---

## 0. Les données du réseau (le vocabulaire à connaître)

| Symbole | Rôle | Dans le code |
|---|---|---|
| `d` | tailles des couches, ex. `[2, 2, 1]` | `self.d` |
| `L` | indice de la dernière couche = `len(d) - 1` | `self.l` |
| `W[l][i][j]` | poids du neurone `i` (couche `l-1`) vers le neurone `j` (couche `l`) | `self.w` |
| `X[l][j]` | sortie du neurone `j` de la couche `l`. **`X[l][0] = 1` = biais** | `self.x` |
| `deltas[l][j]` | signal d'erreur du neurone `j` de la couche `l` | `self.deltas` |

> Le neurone d'indice **0** de chaque couche est le **biais** : sa sortie vaut
> toujours 1. C'est pourquoi les boucles sur les sorties commencent souvent à `j = 1`.

---

## 1. INITIALISATION  `new(d)`

```
POUR chaque couche l de 0 à L :
    SI l == 0 :                      # couche d'entrée : pas de poids entrants
        continuer
    POUR chaque neurone i de 0 à d[l-1] :     # i = 0 -> biais
        POUR chaque neurone j de 0 à d[l] :
            SI j == 0 :  W[l][i][j] = 0        # colonne biais de sortie : inutilisée
            SINON     :  W[l][i][j] = aléatoire dans [-1, 1]

POUR chaque couche l de 0 à L :
    POUR chaque neurone j de 0 à d[l] :
        X[l][j]      = 1 si j==0 sinon 0       # le biais est à 1
        deltas[l][j] = 0
```

**À savoir dire :** « on initialise les poids au hasard dans [-1, 1] pour casser la
symétrie ; si tous les poids étaient égaux, tous les neurones apprendraient la même chose. »

---

## 2. PROPAGATION AVANT  `propagate(entree, classification)`

```
# 1) on place l'entrée dans la couche 0 (après le biais)
POUR j de 1 à d[0] :
    X[0][j] = entree[j-1]

# 2) couche par couche, on calcule chaque neurone
POUR chaque couche l de 1 à L :
    POUR chaque neurone j de 1 à d[l] :
        total = 0
        POUR chaque neurone i de 0 à d[l-1] :         # i=0 -> biais
            total += W[l][i][j] * X[l-1][i]           # somme pondérée
        SI classification OU l < L :                  # tanh sur cachées + sortie si classif
            total = tanh(total)
        X[l][j] = total
```

**Le point clé (question du prof) :**
- **classification** → `tanh` sur **toutes** les couches. Sortie dans `[-1, 1]`, classes **±1**.
- **régression** → `tanh` sur les cachées, mais la **dernière couche reste linéaire**
  (sinon la sortie serait coincée dans `[-1, 1]` et ne pourrait pas valoir 2, 3, ...).

---

## 3. PRÉDICTION  `predict(entree, classification)`

```
propagate(entree, classification)
RENVOYER X[L][1..]        # la couche de sortie, sans le biais
```
Pour **classer** : la classe = **signe** de la sortie (≥ 0 → +1, < 0 → −1).
En **multi-classe** : la classe = indice de la **plus grande** sortie (`argmax`).

---

## 4. ENTRAÎNEMENT  `train(entrees, sorties, steps, lr, classification)`

Descente de gradient **stochastique** : à chaque pas, 1 exemple tiré au hasard.

```
RÉPÉTER steps fois :
    k = indice aléatoire d'un exemple
    propagate(entrees[k], classification)

    # (a) ERREUR DE LA COUCHE DE SORTIE
    POUR j de 1 à d[L] :
        deltas[L][j] = X[L][j] - sorties[k][j-1]        # prédit - attendu
        SI classification :
            deltas[L][j] *= (1 - X[L][j]^2)             # dérivée de tanh

    # (b) RÉTROPROPAGATION vers les couches cachées (de L-1 en descendant)
    POUR l de L à 2 (en descendant) :
        POUR i de 1 à d[l-1] :
            total = 0
            POUR j de 1 à d[l] :
                total += W[l][i][j] * deltas[l][j]      # on remonte l'erreur
            deltas[l-1][i] = total * (1 - X[l-1][i]^2)  # x dérivée de tanh

    # (c) MISE À JOUR DES POIDS
    POUR l de 1 à L :
        POUR i de 0 à d[l-1] :
            POUR j de 1 à d[l] :
                W[l][i][j] -= lr * X[l-1][i] * deltas[l][j]
```

**Les 3 formules à retenir par cœur :**
1. Erreur de sortie : `delta = (prédit − attendu)` ; en classif, `× (1 − X²)`.
2. Erreur cachée : `delta = (Σ W·delta_suivant) × (1 − X²)`.
3. Mise à jour : `W -= lr × entrée × delta`.

> `1 − X²` est la **dérivée de tanh** car `tanh'(x) = 1 − tanh(x)²`. C'est le cœur de
> la rétropropagation : on multiplie par la dérivée pour savoir « dans quel sens » corriger.

---

## 5. tanh vs sigmoid (LA question)

| | sigmoid | **tanh (le nôtre)** |
|---|---|---|
| Sortie | `[0, 1]` | `[-1, 1]` |
| Étiquettes `Y` | `0 / 1` | **`-1 / +1`** |
| Dérivée (delta) | `X·(1 − X)` | **`1 − X²`** |

C'est pourquoi dans le document du prof `Y = [1, -1, -1]` et non `[1, 0, 0]` :
avec tanh, la classe négative se code **−1**.

---

## 6. Quelle architecture pour quel cas de test ?

| Cas de test | Architecture | Mode |
|---|---|---|
| Linear Simple / Multiple | `[2, 1]` | classification |
| **XOR** | `[2, 2, 1]` | classification (300k itér., 2 neurones → converge lentement) |
| **Cross** | `[2, 4, 1]` | classification |
| Multi Linear 3 classes | `[2, 3]` | classification (3 sorties, argmax) |
| Multi Cross | `[2, 32, 32, 3]` | classification (motif difficile) |
| Régressions simples | `[1, 1]` / `[2, 1]` | **régression** (sortie linéaire) |
| Régressions non-linéaires | `[1, 8, 1]` / `[2, 2, 1]` | **régression** |

**Expérimentation à montrer au jury** (« jouez avec les paramètres ») :
- plus d'**itérations** → converge plus sûrement (mais plus lent) ;
- **learning rate** trop grand → instable (ça « saute » le minimum) ;
- plus de **neurones cachés** → frontières plus complexes possibles, mais risque de sur-apprentissage ;
- le MLP est **non-convexe** → selon l'initialisation aléatoire il peut se coincer ;
  on ré-initialise alors et on relance (c'est ce que fait `test_mlp.py`).
```
