# bindings ali
# on charge le .dll rust et on dit a python comment appeler les fonctions

import ctypes
import os

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

    def __del__(self):
        _lib.rbf_free(self._ptr)  # libere la memoire quand python a fini