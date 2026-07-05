# Auteur : Nina
# bindings.py : pont entre Python et Rust (via ctypes) -- MLP uniquement.

import ctypes    # module pour appeler du code compile (C/Rust)
import os        # pour construire le chemin du fichier compile
import platform  # pour detecter le systeme d'exploitation

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

# Correspondance nom d'activation -> code envoye au Rust
_ACTIVATIONS = {"tanh": 0, "sigmoid": 1, "relu": 2}

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

    def __del__(self):
        if getattr(self, "_ptr", None):
            lib.mlp_destroy(self._ptr)  # on libère la mémoire côté Rust
            self._ptr = None
