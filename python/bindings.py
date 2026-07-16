# Auteurs : Valentin BROUC (lineaire, SVM) & Thinina (MLP)
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

# --- ajout (save/load lineaire) ---
lib.linear_export_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.linear_export_weights.restype = None
lib.linear_import_weights.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.linear_import_weights.restype = None
lib.linear_predict_value.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
lib.linear_predict_value.restype = ctypes.c_double
lib.linear_train_regression.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_size_t,
    ctypes.c_size_t,
    ctypes.c_double,
    ctypes.c_size_t,
]
lib.linear_train_regression.restype = None


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

    def fit_regression(self, X, Y, lr=0.01, epochs=1000):
        """Entraine en regression : Y contient des reels, pas des +1/-1."""
        n = len(X)
        plat = [float(v) for exemple in X for v in exemple]
        X_c = (ctypes.c_double * len(plat))(*plat)
        Y_c = (ctypes.c_double * n)(*[float(v) for v in Y])
        lib.linear_train_regression(self._ptr, X_c, Y_c, n, self.input_dim, lr, epochs)

    def predict_value(self, x):
        # valeur brute, avant le signe
        x_c = (ctypes.c_double * len(x))(*x)
        return lib.linear_predict_value(self._ptr, x_c, self.input_dim)

    def get_weights(self):
        n = self.input_dim + 1
        buf = (ctypes.c_double * n)()
        lib.linear_export_weights(self._ptr, buf, n)
        return list(buf)

    def set_weights(self, w):
        buf = (ctypes.c_double * len(w))(*w)
        lib.linear_import_weights(self._ptr, buf, len(w))

    def _etat(self):
        return {"type": "lineaire", "input_dim": self.input_dim, "poids": self.get_weights()}

    @staticmethod
    def _depuis_etat(e):
        m = ModeleLineaire(e["input_dim"])
        m.set_weights(e["poids"])
        return m

    def save_json(self, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self._etat(), f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return ModeleLineaire._depuis_etat(json.load(f))

    def save_binary(self, chemin):
        entete = array.array("i", [self.input_dim])
        poids = array.array("d", self.get_weights())
        with open(chemin, "wb") as f:
            entete.tofile(f); poids.tofile(f)

    @staticmethod
    def load_binary(chemin):
        with open(chemin, "rb") as f:
            dim = array.array("i"); dim.fromfile(f, 1)
            poids = array.array("d"); poids.fromfile(f, dim[0] + 1)
        m = ModeleLineaire(dim[0])
        m.set_weights(list(poids))
        return m

    save = save_json
    load = load_json

    def __del__(self):
        if getattr(self, "_ptr", None):
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

# --- ajout (save/load SVM) ---
lib.svm_nb_samples.argtypes = [ctypes.c_void_p]; lib.svm_nb_samples.restype = ctypes.c_size_t
lib.svm_input_dim.argtypes = [ctypes.c_void_p]; lib.svm_input_dim.restype = ctypes.c_size_t
lib.svm_get_bias.argtypes = [ctypes.c_void_p]; lib.svm_get_bias.restype = ctypes.c_double
lib.svm_get_gamma.argtypes = [ctypes.c_void_p]; lib.svm_get_gamma.restype = ctypes.c_double
lib.svm_export_alphas.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]; lib.svm_export_alphas.restype = None
lib.svm_export_y_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]; lib.svm_export_y_train.restype = None
lib.svm_export_x_train.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]; lib.svm_export_x_train.restype = None
lib.svm_decision_value.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]; lib.svm_decision_value.restype = ctypes.c_double
lib.svm_charger.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double,
                            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t,
                            ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double, ctypes.c_size_t]
lib.svm_charger.restype = ctypes.c_void_p


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

    def decision_value(self, x):
        # valeur continue, avant le signe
        x_c = (ctypes.c_double * len(x))(*x)
        return lib.svm_decision_value(self._ptr, x_c, self.input_dim)

    def _etat(self):
        n = lib.svm_nb_samples(self._ptr)
        dim = lib.svm_input_dim(self._ptr)
        alphas = (ctypes.c_double * n)(); lib.svm_export_alphas(self._ptr, alphas, n)
        ytr = (ctypes.c_double * n)(); lib.svm_export_y_train(self._ptr, ytr, n)
        xtr = (ctypes.c_double * (n * dim))(); lib.svm_export_x_train(self._ptr, xtr, n * dim)
        return {"type": "svm", "n": n, "dim": dim,
                "bias": lib.svm_get_bias(self._ptr), "gamma": lib.svm_get_gamma(self._ptr),
                "alphas": list(alphas), "y_train": list(ytr), "x_train": list(xtr)}

    @staticmethod
    def _depuis_etat(e):
        obj = SVM.__new__(SVM)                 # pas de svm_create : on charge directement
        obj.input_dim = e["dim"]
        a = (ctypes.c_double * len(e["alphas"]))(*e["alphas"])
        xf = (ctypes.c_double * len(e["x_train"]))(*e["x_train"])
        y = (ctypes.c_double * len(e["y_train"]))(*e["y_train"])
        obj._ptr = lib.svm_charger(a, len(e["alphas"]), e["bias"], xf, len(e["x_train"]),
                                   y, e["n"], e["gamma"], e["dim"])
        return obj

    def save_json(self, chemin):
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self._etat(), f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return SVM._depuis_etat(json.load(f))

    def save_binary(self, chemin):
        e = self._etat()
        entete = array.array("i", [e["n"], e["dim"]])
        scal = array.array("d", [e["bias"], e["gamma"]])
        corps = array.array("d", e["alphas"] + e["y_train"] + e["x_train"])
        with open(chemin, "wb") as f:
            entete.tofile(f); scal.tofile(f); corps.tofile(f)

    @staticmethod
    def load_binary(chemin):
        with open(chemin, "rb") as f:
            hdr = array.array("i"); hdr.fromfile(f, 2); n, dim = hdr[0], hdr[1]
            scal = array.array("d"); scal.fromfile(f, 2)
            corps = array.array("d"); corps.fromfile(f, n + n + n * dim)
        return SVM._depuis_etat({"n": n, "dim": dim, "bias": scal[0], "gamma": scal[1],
                                 "alphas": list(corps[:n]), "y_train": list(corps[n:2 * n]),
                                 "x_train": list(corps[2 * n:])})

    save = save_json
    load = load_json

    def __del__(self):
        if getattr(self, "_ptr", None):
            lib.svm_destroy(self._ptr)
            self._ptr = None


# Un-contre-tous : rend multi-classe un modele binaire (lineaire / SVM / RBF).
class UnContreTous:
    """N modeles binaires, un par classe. predict(x) renvoie la liste des scores."""

    def __init__(self, sous_modeles, classes):
        self.sous_modeles = sous_modeles   # liste de modeles binaires (ModeleLineaire ou SVM)
        self.classes = classes             # noms des classes (ex. ["aucun","humain","animal"])

    def predict(self, x, is_classification=True):
        scores = []
        for m in self.sous_modeles:
            if hasattr(m, "decision_value"):     # SVM
                scores.append(m.decision_value(x))
            elif hasattr(m, "predict_value"):    # lineaire
                scores.append(m.predict_value(x))
            else:                                # RBF (predict renvoie deja une valeur continue)
                scores.append(m.predict(x))
        return scores

    @staticmethod
    def entrainer(fabrique, X, y, n_classes, **params):
        """fabrique(input_dim) -> un modele binaire ; entraine 1 modele par classe."""
        X = [list(map(float, r)) for r in X]
        modeles = []
        for c in range(n_classes):
            yb = [1.0 if int(yi) == c else -1.0 for yi in y]   # classe c = +1, le reste = -1
            try:
                m = fabrique(len(X[0]))     # ModeleLineaire(input_dim)
            except TypeError:
                m = fabrique()              # SVM() (ne prend pas d'argument)
            m.fit(X, yb, **params)
            modeles.append(m)
        return UnContreTous(modeles, list(range(n_classes)))

    def save_json(self, chemin):
        data = {"classes": self.classes, "sous_modeles": [m._etat() for m in self.sous_modeles]}
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            data = json.load(f)
        modeles = [UnContreTous._reconstruire(e) for e in data["sous_modeles"]]
        return UnContreTous(modeles, data["classes"])

    @staticmethod
    def _reconstruire(e):
        t = e.get("type")
        if t == "svm":
            return SVM._depuis_etat(e)
        if t == "rbf":
            return RBFNetwork._depuis_etat(e)
        return ModeleLineaire._depuis_etat(e)

    # Sauvegarde binaire : un seul fichier pour les N sous-modeles.
    # entete : 3 int32 [type, nb_modeles, nb_classes], puis chaque sous-modele.
    TYPES = {"lineaire": 0, "svm": 1, "rbf": 2}

    def save_binary(self, chemin):
        etats = [m._etat() for m in self.sous_modeles]
        type_code = UnContreTous.TYPES[etats[0]["type"]]
        with open(chemin, "wb") as f:
            array.array("i", [type_code, len(etats), len(self.classes)]).tofile(f)
            for e in etats:
                if type_code == 0:                                    # lineaire
                    array.array("i", [e["input_dim"]]).tofile(f)
                    array.array("d", e["poids"]).tofile(f)
                elif type_code == 1:                                  # svm
                    array.array("i", [e["n"], e["dim"]]).tofile(f)
                    array.array("d", [e["bias"], e["gamma"]]).tofile(f)
                    array.array("d", e["alphas"] + e["y_train"] + e["x_train"]).tofile(f)
                else:                                                 # rbf
                    nb = len(e["centres"])
                    taille = len(e["centres"][0]) if nb else 0
                    array.array("i", [nb, taille]).tofile(f)
                    array.array("d", [e["gamma"]]).tofile(f)
                    array.array("d", [v for ligne in e["centres"] for v in ligne]
                                + list(e["poids"])).tofile(f)

    @staticmethod
    def load_binary(chemin):
        with open(chemin, "rb") as f:
            entete = array.array("i"); entete.fromfile(f, 3)
            type_code, nb_modeles, n_classes = entete[0], entete[1], entete[2]
            modeles = []
            for _ in range(nb_modeles):
                if type_code == 0:                                    # lineaire
                    dim = array.array("i"); dim.fromfile(f, 1)
                    poids = array.array("d"); poids.fromfile(f, dim[0] + 1)
                    modeles.append(ModeleLineaire._depuis_etat(
                        {"input_dim": dim[0], "poids": list(poids)}))
                elif type_code == 1:                                  # svm
                    hd = array.array("i"); hd.fromfile(f, 2)
                    n, dim = hd[0], hd[1]
                    scal = array.array("d"); scal.fromfile(f, 2)
                    corps = array.array("d"); corps.fromfile(f, n + n + n * dim)
                    modeles.append(SVM._depuis_etat(
                        {"n": n, "dim": dim, "bias": scal[0], "gamma": scal[1],
                         "alphas": list(corps[:n]), "y_train": list(corps[n:2 * n]),
                         "x_train": list(corps[2 * n:])}))
                else:                                                 # rbf
                    hd = array.array("i"); hd.fromfile(f, 2)
                    nb_c, taille = hd[0], hd[1]
                    g = array.array("d"); g.fromfile(f, 1)
                    vals = array.array("d"); vals.fromfile(f, nb_c * taille + nb_c)
                    plat = list(vals[:nb_c * taille])
                    modeles.append(RBFNetwork._depuis_etat({
                        "gamma": g[0],
                        "centres": [plat[i * taille:(i + 1) * taille] for i in range(nb_c)],
                        "poids": list(vals[nb_c * taille:])}))
        return UnContreTous(modeles, list(range(n_classes)))


### =================== Modele MLP / PMC (Thinina) =================== ###
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

    # Adaptateurs (Thinina) : Ali nomme ses methodes get_state / from_state,
    # le lineaire et le SVM utilisent _etat / _depuis_etat -> on aligne les 3.
    def _etat(self):
        e = self.get_state()
        e["type"] = "rbf"
        return e

    @staticmethod
    def _depuis_etat(e):
        return RBFNetwork.from_state(e)

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
