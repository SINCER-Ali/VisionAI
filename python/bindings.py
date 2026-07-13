# Auteurs : Valentin BROUC (lineaire, SVM) & Nina (MLP)
# bindings.py : pont entre Python et Rust (via ctypes)

import ctypes
import os
import platform
import json      # sauvegarde/chargement JSON
import array     # sauvegarde/chargement binaire

# --- Trouver et charger la bibliotheque partagee Rust (UNE seule fois) ---
_dossier = os.path.dirname(__file__)
_racine = os.path.join(_dossier, "..", "lib_rust", "target", "release")

_systeme = platform.system()
if _systeme == "Windows":
    _nom = "lib_rust.dll"
elif _systeme == "Darwin":
    _nom = "lib_rust.dylib"
else:
    _nom = "lib_rust.so"

_chemin = os.path.join(_racine, _nom)
lib = ctypes.CDLL(_chemin)


### =================== Modele Lineaire (Valentin BROUC) =================== ###
lib.linear_create.argtypes = [ctypes.c_size_t]
lib.linear_create.restype = ctypes.c_void_p

lib.linear_train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t,
]
lib.linear_train.restype = None

lib.linear_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.linear_predict.restype = ctypes.c_double

lib.linear_destroy.argtypes = [ctypes.c_void_p]
lib.linear_destroy.restype = None


class ModeleLineaire:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self._ptr = lib.linear_create(input_dim)

    def fit(self, X, Y, lr=0.1, epochs=100):
        n = len(X)
        plat = [v for exemple in X for v in exemple]
        X_c = (ctypes.c_double * len(plat))(*plat)
        Y_c = (ctypes.c_double * n)(*Y)
        lib.linear_train(self._ptr, X_c, Y_c, n, self.input_dim, lr, epochs)

    def predict(self, x):
        x_c = (ctypes.c_double * len(x))(*x)
        return lib.linear_predict(self._ptr, x_c, self.input_dim)

    def __del__(self):
        if self._ptr:
            lib.linear_destroy(self._ptr)
            self._ptr = None


### =================== Modele SVM (Valentin BROUC) =================== ###
lib.svm_create.argtypes = []
lib.svm_create.restype = ctypes.c_void_p

lib.svm_train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t,
    ctypes.c_double,
]
lib.svm_train.restype = None

lib.svm_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.svm_predict.restype = ctypes.c_double

lib.svm_destroy.argtypes = [ctypes.c_void_p]
lib.svm_destroy.restype = None


class SVM:
    def __init__(self):
        self.input_dim = None
        self._ptr = lib.svm_create()

    def fit(self, X, Y, lr=0.001, epochs=1000, gamma=0.0):
        n = len(X)
        self.input_dim = len(X[0])
        plat = [v for ex in X for v in ex]
        X_c = (ctypes.c_double * len(plat))(*plat)
        Y_c = (ctypes.c_double * n)(*Y)
        lib.svm_train(self._ptr, X_c, Y_c, n, self.input_dim, lr, epochs, gamma)

    def predict(self, x):
        x_c = (ctypes.c_double * len(x))(*x)
        return lib.svm_predict(self._ptr, x_c, self.input_dim)

    def __del__(self):
        if self._ptr:
            lib.svm_destroy(self._ptr)
            self._ptr = None


### =================== Modele MLP / PMC (Nina) =================== ###
lib.mlp_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_size_t]
lib.mlp_create.restype = ctypes.c_void_p

_ACTIVATIONS = {"tanh": 0, "sigmoid": 1, "relu": 2}
_ACTIVATIONS_INV = {v: k for k, v in _ACTIVATIONS.items()}

lib.mlp_train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_bool,
]
lib.mlp_train.restype = None

lib.mlp_predict.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_bool,
]
lib.mlp_predict.restype = None

lib.mlp_destroy.argtypes = [ctypes.c_void_p]
lib.mlp_destroy.restype = None

lib.mlp_export_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.mlp_export_weights.restype = None
lib.mlp_import_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.mlp_import_weights.restype = None


class MLP:
    """Enveloppe Python confortable autour du MLP Rust."""

    def __init__(self, npl, activation="tanh"):
        self.npl = list(npl)
        self.n_inputs = self.npl[0]
        self.n_outputs = self.npl[-1]
        self.activation = activation
        code = _ACTIVATIONS.get(activation, 0)
        arr = (ctypes.c_size_t * len(self.npl))(*self.npl)
        self._ptr = lib.mlp_create(arr, len(self.npl), code)

    def fit(self, X, Y, steps=100_000, lr=0.01, is_classification=True):
        n = len(X)
        plat_x = [float(v) for ex in X for v in ex]
        plat_y = [float(v) for yi in Y for v in (yi if hasattr(yi, "__len__") else [yi])]
        X_c = (ctypes.c_double * len(plat_x))(*plat_x)
        Y_c = (ctypes.c_double * len(plat_y))(*plat_y)
        lib.mlp_train(self._ptr, X_c, Y_c, n, self.n_inputs, self.n_outputs,
                      steps, lr, is_classification)

    def predict(self, x, is_classification=True):
        x_c = (ctypes.c_double * len(x))(*x)
        out = (ctypes.c_double * self.n_outputs)()
        lib.mlp_predict(self._ptr, x_c, self.n_inputs, out, self.n_outputs, is_classification)
        return list(out)

    def _nb_poids(self):
        npl = self.npl
        return sum((npl[l - 1] + 1) * (npl[l] + 1) for l in range(1, len(npl)))

    def get_weights(self):
        n = self._nb_poids()
        buf = (ctypes.c_double * n)()
        lib.mlp_export_weights(self._ptr, buf, n)
        return list(buf)

    def set_weights(self, poids):
        buf = (ctypes.c_double * len(poids))(*poids)
        lib.mlp_import_weights(self._ptr, buf, len(poids))

    def save_json(self, chemin):
        data = {"npl": self.npl, "activation": self.activation, "poids": self.get_weights()}
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = MLP(data["npl"], activation=data.get("activation", "tanh"))
        m.set_weights(data["poids"])
        return m

    def save_binary(self, chemin):
        code = _ACTIVATIONS.get(self.activation, 0)
        entete = array.array("i", [len(self.npl)] + list(self.npl) + [code])
        poids = array.array("d", self.get_weights())
        with open(chemin, "wb") as f:
            entete.tofile(f)
            poids.tofile(f)

    @staticmethod
    def load_binary(chemin):
        with open(chemin, "rb") as f:
            nlen = array.array("i"); nlen.fromfile(f, 1)
            npl = array.array("i"); npl.fromfile(f, nlen[0])
            code = array.array("i"); code.fromfile(f, 1)
            npl = list(npl)
            nb = sum((npl[l - 1] + 1) * (npl[l] + 1) for l in range(1, len(npl)))
            poids = array.array("d"); poids.fromfile(f, nb)
        m = MLP(npl, activation=_ACTIVATIONS_INV.get(code[0], "tanh"))
        m.set_weights(list(poids))
        return m

    save = save_json
    load = load_json

    def __del__(self):
        if getattr(self, "_ptr", None):
            lib.mlp_destroy(self._ptr)
            self._ptr = None


# RBF (Ali)
lib.rbf_new.argtypes = [ctypes.c_size_t, ctypes.c_double]
lib.rbf_new.restype = ctypes.c_void_p

lib.rbf_train.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
]
lib.rbf_train.restype = None

lib.rbf_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.rbf_predict.restype = ctypes.c_double

lib.rbf_predict_class.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.rbf_predict_class.restype = ctypes.c_double

lib.rbf_free.argtypes = [ctypes.c_void_p]
lib.rbf_free.restype = None

lib.rbf_gamma.argtypes = [ctypes.c_void_p]
lib.rbf_gamma.restype = ctypes.c_double
lib.rbf_nb_centres.argtypes = [ctypes.c_void_p]
lib.rbf_nb_centres.restype = ctypes.c_size_t
lib.rbf_taille_centre.argtypes = [ctypes.c_void_p]
lib.rbf_taille_centre.restype = ctypes.c_size_t
lib.rbf_export_centres.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.rbf_export_centres.restype = None
lib.rbf_export_poids.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.rbf_export_poids.restype = None
lib.rbf_charger.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double]
lib.rbf_charger.restype = ctypes.c_void_p


def _c_doubles(lst):
    return (ctypes.c_double * len(lst))(*lst)


class RBFNetwork:
    def __init__(self, k, gamma):
        self._ptr = lib.rbf_new(k, gamma)

    def train(self, data, targets, k, iterations=100):
        n = len(data)
        f = len(data[0])
        plat = [float(v) for row in data for v in row]
        lib.rbf_train(self._ptr, _c_doubles(plat), n, f,
                      _c_doubles([float(t) for t in targets]), k, iterations)

    def predict(self, x):
        x = [float(v) for v in x]
        return lib.rbf_predict(self._ptr, _c_doubles(x), len(x))

    def predict_class(self, x):
        x = [float(v) for v in x]
        return lib.rbf_predict_class(self._ptr, _c_doubles(x), len(x))

    def get_state(self):
        # centres + poids + gamma depuis le rust
        nb = lib.rbf_nb_centres(self._ptr)
        taille = lib.rbf_taille_centre(self._ptr)
        c = (ctypes.c_double * (nb * taille))()
        lib.rbf_export_centres(self._ptr, c, nb * taille)
        p = (ctypes.c_double * nb)()
        lib.rbf_export_poids(self._ptr, p, nb)
        centres = [list(c[i * taille:(i + 1) * taille]) for i in range(nb)]
        return {"gamma": lib.rbf_gamma(self._ptr), "centres": centres, "poids": list(p)}

    @staticmethod
    def from_state(s):
        nb = len(s["centres"])
        taille = len(s["centres"][0]) if nb else 0
        plat = [v for ligne in s["centres"] for v in ligne]
        obj = RBFNetwork.__new__(RBFNetwork)
        obj._ptr = lib.rbf_charger(_c_doubles(plat), nb, taille,
                                   _c_doubles(s["poids"]), len(s["poids"]), s["gamma"])
        return obj

    def save_json(self, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return RBFNetwork.from_state(json.load(f))

    def save_binary(self, chemin):
        s = self.get_state()
        nb = len(s["centres"])
        taille = len(s["centres"][0]) if nb else 0
        entete = array.array("i", [nb, taille])
        vals = array.array("d", [s["gamma"]] + [v for ligne in s["centres"] for v in ligne] + list(s["poids"]))
        with open(chemin, "wb") as f:
            entete.tofile(f)
            vals.tofile(f)

    @staticmethod
    def load_binary(chemin):
        with open(chemin, "rb") as f:
            entete = array.array("i"); entete.fromfile(f, 2)
            nb, taille = entete[0], entete[1]
            vals = array.array("d"); vals.fromfile(f, 1 + nb * taille + nb)
        gamma = vals[0]
        plat = list(vals[1:1 + nb * taille])
        poids = list(vals[1 + nb * taille:])
        centres = [plat[i * taille:(i + 1) * taille] for i in range(nb)]
        return RBFNetwork.from_state({"gamma": gamma, "centres": centres, "poids": poids})

    def __del__(self):
        if getattr(self, "_ptr", None):
            lib.rbf_free(self._ptr)
            self._ptr = None
