# Intégration des modèles dans l'API — guide d'équipe

L'API (`server.py`) est conçue pour accueillir facilement les modèles de
chacun. Elle **charge des modèles pré-entraînés depuis le disque** (dossier
`models/`) et les expose au client web.

## Comment ça marche (vue d'ensemble)

```
train_*.py   ->  entraîne + SAUVEGARDE  models/<modele>_weights.json   (hors-ligne)
                        │
server.py    ->  au démarrage : CHARGE chaque modèle depuis models/    (en-ligne)
                        │
             MODELES = { "mlp": <objet>, "lineaire": <objet>, ... }
                        │
  GET  /models   -> liste les noms  -> menu déroulant du client
  POST /predict  -> image + nom du modèle -> classe + scores
```

Chaque modèle du registre `MODELES` doit juste avoir une méthode
`.predict(vecteur)` (liste de scores pour le multi-classe, ou un seul score
pour un modèle binaire — l'API gère les deux).

## Pour AJOUTER ton modèle (linéaire / SVM / RBF)

**1. Côté Rust** (`lib_rust/src/`) : ton module (`linear.rs`, `svm.rs`, `rbf.rs`)
   est déclaré dans `lib.rs` avec ses fonctions `extern "C"` (create/train/predict/destroy).

**2. Côté Python** (`bindings.py`) : ta classe (`ModeleLineaire`, `SVM`, `RBF`)
   existe, avec :
   - `.predict(x)`
   - une méthode de **chargement depuis le disque**, ex. `load_json(chemin)`
     (voir le MLP : `save_json` / `load_json` / `save_binary` / `load_binary`).

**3. Sauvegarde ton modèle entraîné** dans `models/`, ex. `models/svm_weights.json`
   (ton script `train_svm.py` fait le `save_json` à la fin).

**4. Branche-le dans `server.py`** — dans `_charger_modeles()`, ajoute 1 ligne :
```python
from bindings import SVM                                   # en haut du fichier
MODELES["svm"] = SVM.load_json(_chemin_modele("svm_weights.json"))
```

C'est tout. Le menu du client (`/models`) affichera automatiquement `"svm"`.

## Convention des fichiers de modèles

| Modèle | Fichier attendu dans `models/` |
|---|---|
| MLP | `mlp_weights.json` (+ `.bin`) |
| Linéaire | `lineaire_weights.json` |
| SVM | `svm_weights.json` |
| RBF | `rbf_weights.json` |

## Rappel
- Ne pas versionner `target/` ni `.venv/` (déjà dans `.gitignore`).
- Les modèles se **régénèrent** avec les scripts `train_*.py` — décider en équipe
  si on versionne `models/*.json` (lourd) ou si on les gitignore.
