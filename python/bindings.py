# Auteur : Nina
# bindings.py : pont entre Python et Rust (via ctypes) -- MLP uniquement.

import ctypes    # module pour appeler du code compile (C/Rust)
import os        # pour construire le chemin du fichier compile
import platform  # pour detecter le systeme d'exploitation
import json      # sauvegarde/chargement au format JSON (lisible)
import array     # sauvegarde/chargement au format BINAIRE (compact)

# --- Trouver et charger la bibliothèque partagée Rust ---
_dossier = os.path.dirname(__file__)  # dossier de ce script (python/)
_racine = os.path.join(_dossier, "..", "lib_rust", "target", "release")  # sortie de cargo build --release

_systeme = platform.system()
if _systeme == "Windows":
    _nom = "lib_rust.dll"
elif _systeme == "Darwin":   # MacOS
    _nom = "lib_rust.dylib"
else:                        # Linux
    _nom = "lib_rust.so"

_chemin = os.path.join(_racine, _nom)  # chemin complet vers la bibliothèque
lib = ctypes.CDLL(_chemin)             # on charge la DLL compilée


### Signatures des fonctions Rust (types des arguments / retours) ###
# tableau des tailles de couches + longueur + code d'activation (0=tanh,1=sigmoid,2=relu)
lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t]
lib.mlp_create.restype = ctypes.c_void_p                                       # pointeur (ticket)

# Correspondance nom d'activation -> code envoye au Rust (et l'inverse pour le chargement)
_ACTIVATIONS = {"tanh": 0, "sigmoid": 1, "relu": 2}
_ACTIVATIONS_INV = {v: k for k, v in _ACTIVATIONS.items()}  # code -> nom

lib.mlp_train.argtypes = [
    ctypes.c_void_p,                  # pointeur vers le modèle
    ctypes.POINTER(ctypes.c_double),  # x_ptr : entrées aplaties
    ctypes.POINTER(ctypes.c_double),  # y_ptr : sorties attendues aplaties
    ctypes.c_size_t,                  # n_samples
    ctypes.c_size_t,                  # n_inputs  (taille d'une entrée)
    ctypes.c_size_t,                  # n_outputs (taille d'une sortie)
    ctypes.c_size_t,                  # steps
    ctypes.c_double,                  # learning_rate
    ctypes.c_bool,                    # is_classification
]
lib.mlp_train.restype = None

lib.mlp_predict.argtypes = [
    ctypes.c_void_p,                  # pointeur vers le modèle
    ctypes.POINTER(ctypes.c_double),  # entrée
    ctypes.c_size_t,                  # n_inputs
    ctypes.POINTER(ctypes.c_double),  # tampon de sortie (rempli par le Rust)
    ctypes.c_size_t,                  # n_outputs
    ctypes.c_bool,                    # is_classification
]
lib.mlp_predict.restype = None

lib.mlp_destroy.argtypes = [ctypes.c_void_p]
lib.mlp_destroy.restype = None

# export / import des poids (pour sauvegarder/charger un modele)
lib.mlp_export_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.mlp_export_weights.restype = None
lib.mlp_import_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.mlp_import_weights.restype = None


class MLP:
    """Enveloppe Python confortable autour du MLP Rust."""

    def __init__(self, npl, activation="tanh"):
        # npl = architecture, ex. [2, 2, 1] ; activation = "tanh" | "sigmoid" | "relu"
        self.npl = list(npl)
        self.n_inputs = self.npl[0]    # taille d'une entrée  = 1ère couche
        self.n_outputs = self.npl[-1]  # taille d'une sortie  = dernière couche
        self.activation = activation
        code = _ACTIVATIONS.get(activation, 0)              # nom -> code (0 par défaut = tanh)
        arr = (ctypes.c_size_t * len(self.npl))(*self.npl)  # tableau C des tailles
        self._ptr = lib.mlp_create(arr, len(self.npl), code)  # on crée le MLP côté Rust (ticket)

    def fit(self, X, Y, steps=100_000, lr=0.01, is_classification=True):
        n = len(X)  # nombre d'exemples
        # On aplatit X et Y en listes de float. On accepte listes OU numpy,
        # et Y en scalaires (ex. [1, -1]) OU en vecteurs (ex. [[1,-1,-1], ...]).
        plat_x = [float(v) for ex in X for v in ex]
        plat_y = [float(v) for yi in Y for v in (yi if hasattr(yi, "__len__") else [yi])]
        X_c = (ctypes.c_double * len(plat_x))(*plat_x)  # tableau C des entrées
        Y_c = (ctypes.c_double * len(plat_y))(*plat_y)  # tableau C des sorties
        lib.mlp_train(self._ptr, X_c, Y_c, n, self.n_inputs, self.n_outputs,
                      steps, lr, is_classification)     # on lance l'apprentissage côté Rust

    def predict(self, x, is_classification=True):
        x_c = (ctypes.c_double * len(x))(*x)           # entrée -> tableau C
        out = (ctypes.c_double * self.n_outputs)()      # tampon de sortie vide
        lib.mlp_predict(self._ptr, x_c, self.n_inputs, out, self.n_outputs, is_classification)
        return list(out)                                # renvoie la liste des sorties

    # ------- Sauvegarde / chargement du modele (sur disque, en JSON) -------
    def _nb_poids(self):
        # nombre total de poids = somme sur chaque couche de (d[l-1]+1) x (d[l]+1)
        npl = self.npl
        return sum((npl[l - 1] + 1) * (npl[l] + 1) for l in range(1, len(npl)))

    def get_weights(self):
        """Recupere tous les poids du modele (liste plate de float)."""
        n = self._nb_poids()
        buf = (ctypes.c_double * n)()
        lib.mlp_export_weights(self._ptr, buf, n)
        return list(buf)

    def set_weights(self, poids):
        """Remet des poids dans le modele (liste plate de float)."""
        buf = (ctypes.c_double * len(poids))(*poids)
        lib.mlp_import_weights(self._ptr, buf, len(poids))

    # --- Format JSON : lisible (on peut ouvrir le fichier), mais lourd ---
    def save_json(self, chemin):
        """Sauvegarde le modele (archi + activation + poids) en JSON."""
        data = {"npl": self.npl, "activation": self.activation, "poids": self.get_weights()}
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(chemin):
        """Recree un MLP a partir d'un fichier JSON."""
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = MLP(data["npl"], activation=data.get("activation", "tanh"))
        m.set_weights(data["poids"])                    # on remet les poids appris
        return m

    # --- Format BINAIRE : compact et rapide (octets bruts, non lisible) ---
    def save_binary(self, chemin):
        """Sauvegarde le modele en binaire : entete (archi + activation) puis les poids."""
        code = _ACTIVATIONS.get(self.activation, 0)
        entete = array.array("i", [len(self.npl)] + list(self.npl) + [code])  # entiers 32 bits
        poids = array.array("d", self.get_weights())                          # doubles 64 bits
        with open(chemin, "wb") as f:
            entete.tofile(f)                            # on ecrit l'entete
            poids.tofile(f)                             # puis tous les poids

    @staticmethod
    def load_binary(chemin):
        """Recree un MLP a partir d'un fichier binaire."""
        with open(chemin, "rb") as f:
            nlen = array.array("i"); nlen.fromfile(f, 1)          # combien de couches
            npl = array.array("i"); npl.fromfile(f, nlen[0])      # les tailles de couches
            code = array.array("i"); code.fromfile(f, 1)          # le code d'activation
            npl = list(npl)
            nb = sum((npl[l - 1] + 1) * (npl[l] + 1) for l in range(1, len(npl)))  # nb de poids
            poids = array.array("d"); poids.fromfile(f, nb)       # les poids
        m = MLP(npl, activation=_ACTIVATIONS_INV.get(code[0], "tanh"))
        m.set_weights(list(poids))
        return m

    # Alias pratiques : save/load = format JSON par defaut
    save = save_json
    load = load_json

    def __del__(self):
        if getattr(self, "_ptr", None):
            lib.mlp_destroy(self._ptr)  # on libère la mémoire côté Rust
            self._ptr = None
