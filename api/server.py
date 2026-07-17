# Auteur : Thinina
# API REST (FastAPI) : image -> vecteur -> modele (Rust via ctypes) -> classe.
# Lancer depuis api/ :  ..\.venv\Scripts\python.exe -m uvicorn server:app --reload

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

app = FastAPI(title="VisionAI", description="Classe une image : aucun / humain / animal")

# CORS : autorise le client web a appeler l'API depuis une autre origine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# nom du modele -> objet pret a predire ; rempli au demarrage
MODELES = {}


def _chemin_modele(nom):
    return os.path.join(os.path.dirname(__file__), "..", "models", nom)


def _charger_modeles():
    """Charge les modeles PRE-ENTRAINES depuis models/. Un fichier manquant est
    simplement ignore (pas de crash) -> lancer le train_*.py correspondant."""
    MODELES.clear()

    def essayer(nom, fabrique):
        try:
            MODELES[nom] = fabrique()
            print(f"[startup] modele '{nom}' charge")
        except Exception as e:
            print(f"[startup] modele '{nom}' NON charge ({e}) -> lancez son train_*.py")

    # MLP : nativement multi-classe. Les 3 autres : binaires -> un-contre-tous.
    essayer("mlp", lambda: MLP.load_json(_chemin_modele("mlp_weights.json")))
    essayer("lineaire", lambda: UnContreTous.load_json(_chemin_modele("lineaire_weights.json")))
    essayer("svm", lambda: UnContreTous.load_json(_chemin_modele("svm_weights.json")))
    essayer("rbf", lambda: UnContreTous.load_json(_chemin_modele("rbf_weights.json")))
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
    """Modeles disponibles -> alimente le menu du client web."""
    return {"modeles": list(MODELES.keys())}


@app.post("/reload-models")
def reload_models():
    """Recharge les modeles depuis le disque sans redemarrer le serveur.
    Utile apres avoir (re)lance un train_*.py pendant que l'API tourne."""
    _charger_modeles()
    return {"status": "recharge", "modeles": list(MODELES.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("mlp")):
    """Recoit une image + le nom du modele choisi, renvoie la classe + les scores."""
    if model not in MODELES:
        raise HTTPException(404, f"Modele '{model}' inconnu. Dispo : {list(MODELES.keys())}")
    data = await file.read()
    try:
        vec = image_to_vector(io.BytesIO(data)).tolist()
    except Exception as e:
        raise HTTPException(400, f"Image illisible : {e}")
    import math
    sorties = MODELES[model].predict(vec)
    sorties = list(sorties) if isinstance(sorties, (list, tuple)) else [float(sorties)]
    sorties = [v if math.isfinite(v) else 0.0 for v in sorties]   # inf/nan -> 0 : pas de crash JSON
    if len(sorties) >= len(CLASSES):                       # multi-classe -> argmax
        idx = int(np.argmax(sorties[:len(CLASSES)]))
        scores = {CLASSES[i]: round(float(sorties[i]), 3) for i in range(len(CLASSES))}
    else:                                                  # binaire -> signe
        idx = 0 if sorties[0] >= 0 else 1
        scores = {"score": round(float(sorties[0]), 3)}
    return {"modele": model, "classe": CLASSES[idx], "scores": scores}
