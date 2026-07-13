# bindings ali
# on charge le .dll rust et on dit a python comment appeler les fonctions

import ctypes
import os
import json
import array

# on trouve le .dll et on le charge
_dll_path = os.path.join(os.path.dirname(__file__), '..', 'lib_rust', 'target', 'debug', 'lib_rust.dll')
_lib = ctypes.CDLL(os.path.abspath(_dll_path))

# rbf_new : cree le modele et retourne son adresse
_lib.rbf_new.argtypes = [ctypes.c_size_t, ctypes.c_double]
_lib.rbf_new.restype  = ctypes.c_void_p

# rbf_train : on donne les donnees au modele pour quil apprenne
_lib.rbf_train.argtypes = [
    ctypes.c_void_p,                  # adresse du modele
    ctypes.POINTER(ctypes.c_double),  # les entrees en tableau plat
    ctypes.c_size_t,                  # nb d exemples
    ctypes.c_size_t,                  # taille d un exemple
    ctypes.POINTER(ctypes.c_double),  # les sorties attendues
    ctypes.c_size_t,                  # k
    ctypes.c_size_t,                  # iterations
]
_lib.rbf_train.restype = None  # retourne rien

# rbf_predict : retourne une valeur predite
_lib.rbf_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.rbf_predict.restype  = ctypes.c_double

# rbf_predict_class : retourne +1 ou -1
_lib.rbf_predict_class.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.rbf_predict_class.restype  = ctypes.c_double

# rbf_free : supprime le modele de la memoire
_lib.rbf_free.argtypes = [ctypes.c_void_p]
_lib.rbf_free.restype  = None

# fonctions pour sauvegarder / recharger le modele
_lib.rbf_gamma.argtypes = [ctypes.c_void_p]
_lib.rbf_gamma.restype  = ctypes.c_double
_lib.rbf_nb_centres.argtypes = [ctypes.c_void_p]
_lib.rbf_nb_centres.restype  = ctypes.c_size_t
_lib.rbf_taille_centre.argtypes = [ctypes.c_void_p]
_lib.rbf_taille_centre.restype  = ctypes.c_size_t
_lib.rbf_export_centres.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.rbf_export_centres.restype  = None
_lib.rbf_export_poids.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
_lib.rbf_export_poids.restype  = None
_lib.rbf_charger.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_size_t,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_size_t, ctypes.c_double]
_lib.rbf_charger.restype  = ctypes.c_void_p


def to_c_array(lst):
    # convertit une liste python en tableau que le C comprend
    return (ctypes.c_double * len(lst))(*lst)


class RBFNetwork:
    def __init__(self, k, gamma):
        self._ptr = _lib.rbf_new(k, gamma)  # cree le modele

    def train(self, data, targets, k, iterations=100):
        n_samples  = len(data)
        n_features = len(data[0])
        flat = [x for row in data for x in row]  # aplatit les listes en un seul tableau
        _lib.rbf_train(self._ptr, to_c_array(flat), n_samples, n_features, to_c_array(targets), k, iterations)

    def predict(self, x):
        return _lib.rbf_predict(self._ptr, to_c_array(x), len(x))

    def predict_class(self, x):
        return _lib.rbf_predict_class(self._ptr, to_c_array(x), len(x))

    def get_state(self):
        # recupere centres + poids + gamma depuis le rust
        nb = _lib.rbf_nb_centres(self._ptr)
        taille = _lib.rbf_taille_centre(self._ptr)
        c = (ctypes.c_double * (nb * taille))()
        _lib.rbf_export_centres(self._ptr, c, nb * taille)
        p = (ctypes.c_double * nb)()
        _lib.rbf_export_poids(self._ptr, p, nb)
        centres = [list(c[i * taille:(i + 1) * taille]) for i in range(nb)]
        return {"gamma": _lib.rbf_gamma(self._ptr), "centres": centres, "poids": list(p)}

    @staticmethod
    def from_state(s):
        # recree un rbf a partir d'un etat sauvegarde (sans reentrainer)
        nb = len(s["centres"])
        taille = len(s["centres"][0]) if nb else 0
        plat = [v for ligne in s["centres"] for v in ligne]
        obj = RBFNetwork.__new__(RBFNetwork)
        obj._ptr = _lib.rbf_charger(to_c_array(plat), nb, taille,
                                    to_c_array(s["poids"]), len(s["poids"]), s["gamma"])
        return obj

    def save_json(self, chemin):
        # format lisible
        with open(chemin, "w", encoding="utf-8") as f:
            json.dump(self.get_state(), f)

    @staticmethod
    def load_json(chemin):
        with open(chemin, "r", encoding="utf-8") as f:
            return RBFNetwork.from_state(json.load(f))

    def save_binary(self, chemin):
        # format compact : entete (nb centres, taille) puis gamma + centres + poids
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
        _lib.rbf_free(self._ptr)  # libere la memoire quand python a fini