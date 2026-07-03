# Auteur : Valentin BROUC
# bindings.py : pont entre Python et Rust (via ctypes)

import ctypes # module pour appeler du code compile (C/Rust)
import os # pour construire le chemin du fichier compile
import platform # pour detecter le systeme d'exploitation

# Trouver et charger la bibliothèque partagée Rust
_dossier = os.path.dirname(__file__) # dossier où on trouve ce script (python/)
_racine = os.path.join(_dossier, "..", "lib_rust", "target", "release") # sortie de "cargo build --release"

_systeme = platform.system() # detecter le systeme d'exploitation
if _systeme == "Windows":
    _nom = "lib_rust.dll"
elif _systeme == "Darwin": # MacOS
    _nom = "lib_rust.dylib"
else: # Linux
    _nom = "lib_rust.so"


_chemin = os.path.join(_racine, _nom) # chemin complet vers la bibliothèque partagée
lib = ctypes.CDLL(_chemin) # charger la bibliothèque partagée


### Pour modèle Linéaire ###
# Déclarer les signatures (types) de chaque fonction Rust que l'on veut utiliser depuis Python
lib.linear_create.argtypes = [ctypes.c_size_t] # entree : input_dim (un entier "usize" en Rust)
lib.linear_create.restype = ctypes.c_void_p # sortie : pointeur vers le modèle (le "ticket")

lib.linear_train.argtypes = [
    ctypes.c_void_p, # pointeur vers le modèle
    ctypes.POINTER(ctypes.c_double), # x_ptr : adresse du tableau des entrées (features)
    ctypes.POINTER(ctypes.c_double), # y_ptr : adresse du tableau des sorties (targets)
    ctypes.c_size_t, # taille du tableau (n_samples)
    ctypes.c_size_t, # input_dim (dimension des features)
    ctypes.c_double, # learning_rate
    ctypes.c_size_t, # epochs
]
lib.linear_train.restype = None # linear_train ne retourne rien

lib.linear_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t] # entrée : pointeur vers le modèle, adresse du tableau des features, taille du tableau
lib.linear_predict.restype = ctypes.c_double # renvoie un double (+1.0 ou -1.0)

lib.linear_destroy.argtypes = [ctypes.c_void_p] # entrée : pointeur à libérer
lib.linear_destroy.restype = None # linear_destroy ne retourne rien

# Classe Python confortable autour du pointeur

class ModeleLineaire:
    def __init__(self, input_dim):
        self.input_dim = input_dim # on retient la taille d'une entrée
        self._ptr = lib.linear_create(input_dim) # on crée le modèle en Rust et on retient le pointeur (récupère le ticket)

    def fit(self, X, Y, lr=0.1, epochs=100):
        n = len(X) # nombre d'échantillons
        plat = [v for exemple in X for v in exemple] # aplatit X : [[1,2],[3,4]] -> [1,2,3,4]
        X_c = (ctypes.c_double * len(plat))(*plat) # on transforme la liste en tableau C de doubles
        Y_c = (ctypes.c_double * n)(*Y) # idem pour les étiquettes
        lib.linear_train(self._ptr, X_c, Y_c, n, self.input_dim, lr, epochs) # on lance l'apprentissage côté Rust

    def predict(self, x):
        x_c = (ctypes.c_double * len(x))(*x) # on transforme le point x en tableau C
        return lib.linear_predict(self._ptr, x_c, self.input_dim) # on appelle la fonction Rust et on retourne la classe prédite (+1.0 ou -1.0)
    
    def __del__(self):
        if self._ptr:
            lib.linear_destroy(self._ptr) # on libère la mémoire côté Rust
            self._ptr = None # on ne garde plus le pointeur

### Fin Modèle Linéaire ###

### Pour Modèle SVM ###
# Auteur: Valentin BROUC

lib.svm_create.argtypes = [] # svm_create ne prend aucun argument
lib.svm_create.restype = ctypes.c_void_p # on renvoie un pointeur (ticket)

lib.svm_train.argtypes = [ # types des arguments pour svm_train
    ctypes.c_void_p, # pointeur vers le modèle
    ctypes.POINTER(ctypes.c_double), # x_ptr : adresse du tableau des entrées (features)
    ctypes.POINTER(ctypes.c_double), # y_ptr : adresse du tableau des sorties (targets)
    ctypes.c_size_t, # taille du tableau (n_samples)
    ctypes.c_size_t, # input_dim (dimension des features)
    ctypes.c_double, # learning_rate
    ctypes.c_size_t, # epochs
    ctypes.c_double, # gamma (0 = linéaire, >0 = RBF)
]
lib.svm_train.restype = None # svm_train ne retourne rien

lib.svm_predict.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t] # entrée : pointeur vers le modèle, adresse du tableau des features, taille du tableau
lib.svm_predict.restype = ctypes.c_double # renvoie un double (+1.0 ou -1.0)

lib.svm_destroy.argtypes = [ctypes.c_void_p] # entrée : pointeur à libérer
lib.svm_destroy.restype = None # svm_destroy ne retourne rien

class SVM:
    def __init__(self):
        self.input_dim = None # on remplira ce champ au moment du fit
        self._ptr = lib.svm_create() # on créé le modèle côté Rust, donc on récupère le ticket
        
    def fit(self, X, Y, lr=0.001, epochs=1000, gamma=0.0): # lance l'entrainement ; gamma=0 -> noyau linéaire
        n = len(X) # nombre d'exemples
        self.input_dim = len(X[0]) # taille d'un exemple (déduite du 1er point)
        plat = [v for ex in X for v in ex] # on aplatit X : [[1,3],[2,1]] -> [1,3,2,1]
        X_c = (ctypes.c_double * len(plat))(*plat) # on transforme la liste en tableau C de doubles
        Y_c = (ctypes.c_double * n)(*Y) # idem pour les classes
        lib.svm_train(self._ptr, X_c, Y_c, n, self.input_dim, lr, epochs, gamma) # on lance l'entraînement côté Rust

    def predict(self, x): # prédit la classe d'un point x
        x_c = (ctypes.c_double * len(x))(*x) # transforme le point x en tableau C
        return lib.svm_predict(self._ptr, x_c, self.input_dim) # renvoie +1.0 / -1.0
    
    def __del__(self): # appelé quand l'objet Python disparait
        if self._ptr: # si le modèle existe encore
            lib.svm_destroy(self._ptr) # on rend le ticket donc libère la mémoire Rust
            self._ptr = None # évite une double libération

### Fin du SVM ###