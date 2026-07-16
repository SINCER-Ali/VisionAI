# Auteur : Thinina
# api/server.py : API REST (FastAPI) pour le client web.
# Fonctions : uploader une image, CHOISIR le modele, predire ; lister les modeles.
# Chaine : image -> vecteur (preprocessing) -> modele (Rust via ctypes) -> classe.
#
# Lancer :  ..\.venv\Scripts\python.exe -m uvicorn server:app --reload
#           (depuis api/ ; interface : http://127.0.0.1:8000/docs)

import io
import os
import sys
import json

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Rendre le dossier python/ importable (bindings.py + preprocessing.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from bindings import MLP, ModeleLineaire, SVM, UnContreTous, RBFNetwork  # noqa: F401
from preprocessing import image_to_vector

CLASSES = ["aucun", "humain", "animal"]  # aucun=0, humain=1, animal=2
CACHEE = 32
STEPS = 30_000
LR = 0.01

app = FastAPI(title="VisionAI", description="Classe une image : aucun / humain / animal")

# CORS : autorise le client web (navigateur) a appeler l'API depuis une autre origine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Registre des modeles disponibles (nom -> objet modele pret a predire).
# Rempli au demarrage. C'est ce que le menu du client web affichera.
MODELES = {}


def _chemin_dataset(nom):
    return os.path.join(os.path.dirname(__file__), "..", "datasets", nom)


def _one_hot_pm1(y, n_classes):
    Y = -np.ones((len(y), n_classes))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _chemin_modele(nom):
    return os.path.join(os.path.dirname(__file__), "..", "models", nom)


def _charger_mlp():
    """Charge le MLP PRE-ENTRAINE depuis le disque (demarrage instantane).
    Si aucun modele n'est sauvegarde, on entraine + sauvegarde (secours)."""
    chemin = _chemin_modele("mlp_weights.json")
    if os.path.exists(chemin):
        print(f"[startup] chargement du modele pre-entraine : {chemin}")
        return MLP.load_json(chemin)                     # <-- charge depuis le disque (JSON)
    # Secours : pas encore de modele -> entrainement (mieux : lancer train_mlp.py avant)
    print("[startup] aucun modele sauvegarde -> entrainement (lancez plutot train_mlp.py)")
    X = np.load(_chemin_dataset("X_train.npy"))
    y = np.load(_chemin_dataset("y_train.npy"))
    m = MLP([X.shape[1], CACHEE, len(CLASSES)])
    m.fit(X, _one_hot_pm1(y, len(CLASSES)), steps=STEPS, lr=LR, is_classification=True)
    m.save_json(chemin)
    return m


def _charger_modeles():
    """(Re)charge TOUS les modeles pre-entraines depuis le disque (models/).
    Chaque entree : "nom" -> objet avec une methode .predict(vecteur).
    Un modele dont le fichier manque est simplement ignore (pas de crash)."""
    MODELES.clear()

    def essayer(nom, fabrique):
        try:
            MODELES[nom] = fabrique()
            print(f"[startup] modele '{nom}' charge")
        except Exception as e:
            print(f"[startup] modele '{nom}' NON charge ({e}) -> lancez son train_*.py")

    def _charger_rbf():
        # RBF d'Ali : rbf_weights.json = {"classes":..., "modeles":[etats]} (un-contre-tous)
        with open(_chemin_modele("rbf_weights.json"), encoding="utf-8") as f:
            data = json.load(f)
        modeles = [RBFNetwork.from_state(s) for s in data["modeles"]]
        return UnContreTous(modeles, data["classes"])

    essayer("mlp", _charger_mlp)   # MLP (Thinina)
    essayer("lineaire", lambda: UnContreTous.load_json(_chemin_modele("lineaire_weights.json")))  # Valentin (un-contre-tous)
    essayer("svm", lambda: UnContreTous.load_json(_chemin_modele("svm_weights.json")))            # Valentin (un-contre-tous)
    essayer("rbf", _charger_rbf)                                                                    # Ali (un-contre-tous)
    print(f"[startup] modeles disponibles : {list(MODELES.keys())}")


@app.on_event("startup")
def au_demarrage():
    _charger_modeles()


@app.get("/health")
def health():
    """Verifie que le serveur tourne."""
    return {"status": "ok" if MODELES else "aucun modele", "classes": CLASSES}


@app.get("/models")
def liste_modeles():
    """Liste des modeles disponibles -> alimente le menu deroulant du client web."""
    return {"modeles": list(MODELES.keys())}


@app.post("/reload-models")
def reload_models():
    """Recharge/reentraine les modeles sans redemarrer le serveur."""
    _charger_modeles()
    return {"status": "recharge", "modeles": list(MODELES.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("mlp")):
    """Recoit une image + le nom du modele choisi, renvoie la classe + les scores."""
    if model not in MODELES:
        raise HTTPException(404, f"Modele '{model}' inconnu. Dispo : {list(MODELES.keys())}")
    data = await file.read()
    try:
        vec = image_to_vector(io.BytesIO(data)).tolist()   # image -> vecteur 12288
    except Exception as e:
        raise HTTPException(400, f"Image illisible : {e}")
    import math
    sorties = MODELES[model].predict(vec)
    # Robuste aux 2 familles de modeles :
    #  - MLP / RBF  -> renvoient une LISTE de scores (un par classe) -> argmax
    #  - lineaire / SVM (binaires) -> renvoient UN seul score -> signe
    sorties = list(sorties) if isinstance(sorties, (list, tuple)) else [float(sorties)]
    # securite : on remplace inf/nan (modele divergent) par 0 -> pas de crash JSON
    sorties = [v if math.isfinite(v) else 0.0 for v in sorties]
    if len(sorties) >= len(CLASSES):                       # modele multi-classe
        idx = int(np.argmax(sorties[:len(CLASSES)]))
        scores = {CLASSES[i]: round(float(sorties[i]), 3) for i in range(len(CLASSES))}
    else:                                                  # modele binaire (+1 / -1)
        idx = 0 if sorties[0] >= 0 else 1
        scores = {"score": round(float(sorties[0]), 3)}
    return {"modele": model, "classe": CLASSES[idx], "scores": scores}
