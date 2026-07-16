# VisionAI — Classification d'images par Machine Learning

Projet Annuel 3A Big Data — **2025-2026**

Application qui classe une photo en **3 catégories** : `aucun` / `humain` / `animal`.
Les **4 modèles de Machine Learning sont implémentés from scratch en Rust** (aucune
bibliothèque de ML), exposés à Python via **ctypes**, servis par une **API REST** et
utilisables depuis un **site web**.

---

## Équipe et répartition

| Modèle | Auteur | Fichier Rust |
|---|---|---|
| **Modèle linéaire** (perceptron de Rosenblatt) | Valentin BROUC | `lib_rust/src/linear.rs` |
| **SVM** (dual + noyau RBF) | Valentin BROUC | `lib_rust/src/svm.rs` |
| **MLP / PMC** (perceptron multi-couches) | Thinina | `lib_rust/src/mlp.rs` |
| **RBF Network** (k-means + moindres carrés) | Ali | `lib_rust/src/rbf.rs` |

Intégration (API, save/load, un-contre-tous, preprocessing) : **Thinina**.

---

## Architecture

```
  Rust (lib_rust/)                Python (python/)              Web
  ┌──────────────────┐            ┌─────────────────┐      ┌──────────────┐
  │ linear.rs        │            │ bindings.py     │      │ web_client/  │
  │ svm.rs           │  cdylib    │  (ctypes)       │      │ index.html   │
  │ mlp.rs           │ ─────────► │ preprocessing.py│ ───► │              │
  │ rbf.rs           │  lib_rust  │ train_*.py      │ API  │  fetch()     │
  │ lib.rs (extern C)│    .dll    │ test_*.py       │      │              │
  └──────────────────┘            └─────────────────┘      └──────────────┘
                                          │
                                     models/*.json|bin  ──►  api/server.py
```

**Pourquoi ctypes et pas PyO3 ?** L'ABI C est le standard le plus simple et le plus
portable ; c'est aussi l'approche de l'exemple d'interopérabilité fourni en cours.
Aucune dépendance lourde (ni maturin, ni build backend).

### Structure du dépôt

| Dossier | Contenu |
|---|---|
| `lib_rust/` | les 4 modèles en Rust + `lib.rs` (façade `extern "C"`) |
| `python/` | `bindings.py` (pont ctypes), `preprocessing.py`, `train_*.py`, `test_*.py` |
| `api/` | `server.py` (FastAPI) — charge les modèles pré-entraînés depuis `models/` |
| `web_client/` | `index.html` — site statique qui appelle l'API |
| `notebook/` | `models_cas_test.ipynb` (cas de tests du prof), `experimentation_models.ipynb` |
| `datasets/` | `train/{aucun,humain,animal}/` (images) et les `.npy` générés |
| `models/` | modèles entraînés (JSON + binaire) — **gitignorés**, régénérés par `train_*.py` |
| `rapport/` | le rapport |
| `Syllabus_Informations/` | le matériel officiel du prof (**ne pas modifier**) |

---

## Les données

- **750 photos** : **250 par classe** (dataset **équilibré**)
- Prétraitement : RGB → **64 × 64** → aplati → normalisé dans [0,1] → **12288 entrées**
- Découpage **80 / 20 stratifié** (proportions conservées par classe) → **600 train / 150 test**
- Reproductible : `np.random.seed(42)` dans `preprocessing.py`
- Formats lus : `.jpg .jpeg .png .bmp .webp .heic` (le HEIC des iPhone via `pillow-heif`)

> **Baseline = 33.3 %** : le score d'un modèle qui répondrait toujours la même classe.
> C'est le repère — tout modèle en dessous n'a rien appris.

---

## Prérequis

- **Python 3.11.7** (environnement virtuel `.venv/` à la racine)
- **Rust / cargo 1.93**

```powershell
# Installer les dependances Python (depuis la racine du projet)
.venv\Scripts\python.exe -m pip install -r api\requirements.txt
```

---

## Lancer le projet

> ### ⚠️ Règle d'or : **compiler le Rust AVANT d'ouvrir Jupyter ou l'API**
> Dès qu'un processus Python fait `import bindings`, il **verrouille `lib_rust.dll`**
> et `cargo build` échoue avec `Access is denied`. Ferme Jupyter (**File → Shut Down**)
> et l'API (**Ctrl+C**) avant de recompiler.

### 1. Compiler la bibliothèque Rust

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\lib_rust
cargo build --release
```
Produit `lib_rust/target/release/lib_rust.dll` (les 4 modèles dans **une seule** DLL).

Vérifier que Python la charge bien :
```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\python
..\.venv\Scripts\python.exe -c "import bindings; print('OK - DLL chargee')"
```

### 2. Préparer les données

> Uniquement quand les **images changent** (ajout/suppression) ou que `TAILLE` /
> `PART_TEST` sont modifiés. Sinon les `.npy` existants suffisent.

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\python
..\.venv\Scripts\python.exe preprocessing.py
```
Génère `datasets/X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`.

### 3. Entraîner les 4 modèles (hors ligne)

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\python
..\.venv\Scripts\python.exe train_mlp.py
..\.venv\Scripts\python.exe train_linear.py
..\.venv\Scripts\python.exe train_svm.py
..\.venv\Scripts\python.exe train_rbf.py
```
Chaque script écrit **2 formats** dans `models/` : `<modele>_weights.json` (lisible) et
`<modele>_weights.bin` (compact, ~2.5× plus léger), et affiche précision **train / test**,
**détail par classe**, **durée d'entraînement** et **baseline**.

### 4. Lancer l'API (terminal 1 — laisser tourner)

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\api
..\.venv\Scripts\python.exe -m uvicorn server:app --reload
```
Au démarrage tu dois voir :
```
[startup] modele 'mlp' charge
[startup] modele 'lineaire' charge
[startup] modele 'svm' charge
[startup] modele 'rbf' charge
[startup] modeles disponibles : ['mlp', 'lineaire', 'svm', 'rbf']
```
Documentation interactive : **http://127.0.0.1:8000/docs**

### 5. Ouvrir le site web

Double-cliquer sur **`web_client/index.html`** (ou l'ouvrir dans Chrome).
L'API doit tourner en parallèle (étape 4) ; le champ *Serveur API* pointe sur
`http://127.0.0.1:8000`. Le CORS est autorisé côté serveur.

### 6. Ouvrir les notebooks (terminal 2)

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI
.venv\Scripts\python.exe -m jupyter lab
```

---

## Tester

Un `test_<modele>.py` par modèle — vérifications rapides, sans dataset d'images :

```powershell
cd C:\Users\thini\Desktop\PA\VisionAI\python
..\.venv\Scripts\python.exe test_linear.py    # 6/6 bien classes
..\.venv\Scripts\python.exe test_svm.py       # Lineaire 6/6, XOR (RBF) 4/4
..\.venv\Scripts\python.exe test_rbf.py       # 4/4 sur XOR
..\.venv\Scripts\python.exe test_mlp.py       # Linear Simple, XOR, multi-classe, regression
```

---

## L'API

| Route | Méthode | Rôle |
|---|---|---|
| `/health` | GET | état du serveur + liste des classes |
| `/models` | GET | modèles disponibles → alimente le menu du site |
| `/predict` | POST | image (`file`) + `model` → classe prédite + scores |
| `/reload-models` | POST | recharge les modèles depuis le disque sans redémarrer |
| `/docs` | GET | documentation interactive (Swagger) |

Exemple :
```powershell
curl.exe http://127.0.0.1:8000/models
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@chemin\image.jpg" -F "model=mlp"
```

> L'API **ne ré-entraîne rien** : elle **charge les modèles pré-entraînés** depuis `models/`,
> conformément au syllabus. Un modèle dont le fichier manque est simplement ignoré
> (le serveur ne plante pas) — lancer son `train_*.py` pour le générer.

---

## Les notebooks

| Notebook | Contenu |
|---|---|
| **`models_cas_test.ipynb`** | Les **11 cas de tests du prof** (données **verbatim** de son fichier), passés au **linéaire**, au **MLP** et au **RBF**. Frontières de décision, surfaces de régression, et un tableau comparant nos résultats à **ses annotations OK/KO**. |
| **`experimentation_models.ipynb`** | **Expérimentation** sur nos vraies images : sous/sur-apprentissage, effet de chaque hyperparamètre (courbes Matplotlib), comparaison des 4 modèles. |

> Le notebook d'expérimentation utilise **exactement les hyperparamètres de production**
> (ceux des `train_*.py`) : ses chiffres sont donc **identiques** à ceux des scripts.

---

## Hyperparamètres de production

| Modèle | Réglages | Fichier |
|---|---|---|
| **MLP** | `[12288, 32, 3]`, `steps=50 000`, `lr=0.01`, activation `tanh` | `train_mlp.py` |
| **Linéaire** | `lr=0.01`, `epochs=50`, un-contre-tous (3 modèles) | `train_linear.py` |
| **SVM** | `lr=0.001`, `epochs=300`, `gamma=0.001` (noyau RBF), `MAX_EXEMPLES=60` | `train_svm.py` |
| **RBF** | `k=20` centres, `gamma=0.01`, `iters=15` (k-means) | `train_rbf.py` |

**Pourquoi `MAX_EXEMPLES=60` pour le SVM ?** Un SVM à noyau **conserve tous ses exemples
d'entraînement** (`f(x) = biais + Σ αₙ·yₙ·K(xₙ,x)`) → mémoire en O(n·d) et entraînement
en O(n²). Sur des images à 12288 dimensions, le modèle devient vite ingérable. Le compromis
précision/temps est mesuré dans `experimentation_models.ipynb`.

**Multi-classe** : le MLP est **nativement** multi-classe (3 neurones de sortie + argmax).
Le linéaire, le SVM et le RBF sont **binaires** → combinés par **un-contre-tous**
(1 modèle par classe, puis argmax des scores continus) — le *« Linear Model x3 »* du cours,
implémenté par la classe `UnContreTous` de `bindings.py`.

---

## Notes techniques

- **`unsafe` en Rust** : confiné à `lib.rs` (la frontière FFI), pour déréférencer les
  pointeurs bruts venant de Python. La logique des modèles est 100 % sûre.
- **Mémoire** : `Box::into_raw` transmet le modèle à Python, `Box::from_raw` le libère
  dans `*_destroy` ; côté Python, `__del__` s'en charge automatiquement.
- **Non-convexité du MLP** : sur les cas type XOR avec l'architecture minimale `[2,2,1]`,
  la descente de gradient se coince dans un minimum local selon l'initialisation aléatoire
  (~1 fois sur 2). Les tests et notebooks **réinitialisent** jusqu'à convergence —
  traitement standard d'une optimisation non convexe.
- **Fichiers ignorés par Git** : `lib_rust/target/`, `models/*.json|bin`, `datasets/train/`
  (trop lourds ; tout est régénéré par `preprocessing.py` puis `train_*.py`).
