// Auteur : Valentin BROUC
// Facade "C" expose le modèle linéaire pour l'utiliser depuis du code C

mod linear; // Importe le module linear.rs
use linear::LinearModel; // Utilise la structure LinearModel du module linear.rs
use std::slice; // Importe le module slice pour manipuler des pointeurs et des slices

#[no_mangle] // Indique au compilateur de ne pas modifier le nom de la fonction pour l'exporter
pub extern "C" fn linear_create(input_dim: usize) -> *mut LinearModel { // Crée un modèle linéaire et retourne un pointeur vers celui-ci
    Box::into_raw(Box::new(LinearModel::new(input_dim))) // Place le modele dans un Box (allocation sur le tas) et retourne un pointeur brut vers celui-ci
} // Fin de la fonction linear_create

#[no_mangle] // Indique au compilateur de ne pas modifier le nom de la fonction pour l'exporter
pub extern "C" fn linear_train( // Entraine le modèle linéaire sur un ensemble d'exemples
    model_ptr: *mut LinearModel, // Pointeur vers le modèle linéaire à entraîner
    x_ptr: *const f64, // Pointeur vers le tableau contenant toutes les entrées (chaque entrée est un vecteur aplati)
    y_ptr: *const f64, // Pointeur vers le tableau contenant toutes les classes attendues (1 ou -1) pour chaque entrée correspondante
    n_samples: usize, // Nombre d'exemples d'apprentissage
    input_dim: usize, // Taille d'un exemple (nombre d'entrées)
    lr: f64, // Taux d'apprentissage (learning rate) pour la mise à jour des poids
    epochs: usize // Nombre d'itérations sur l'ensemble des exemples pour l'apprentissage
) {
    let model = unsafe { &mut *model_ptr }; // Récupère depuis le pointeur brut le modèle linéaire (unsafe car on manipule des pointeurs bruts)
    let all_x = unsafe { slice::from_raw_parts(x_ptr, n_samples * input_dim) }; // Reconstruit le tableau des entrées à partir du pointeur brut + sa longueur
    let all_y = unsafe { slice::from_raw_parts(y_ptr, n_samples) }; // Reconstruit le tableau des classes attendues
    model.train(all_x, all_y, n_samples, input_dim, lr, epochs); // Appelle la méthode train du modèle linéaire avec les données reconstruites
} // Fin de la fonction linear_train

#[no_mangle] // Indique au compilateur de ne pas modifier le nom de la fonction pour l'exporter
pub extern "C" fn linear_predict(model_ptr:*mut LinearModel, x_ptr: *const f64, input_dim: usize) -> f64 { // Prédit la classe d'un point
    let model = unsafe{&*model_ptr}; // Récupère depuis le pointeur brut le modèle linéaire (unsafe car on manipule des pointeurs bruts)
    let x = unsafe{slice::from_raw_parts(x_ptr, input_dim)}; // Reconstruit le tableau de l'entrée à partir du pointeur brut
    model.predict_class(x) // Appelle la méthode predict_class du modèle linéaire avec l'entrée reconstruite et retourne la classe prédite
} // fin de la fonction linear_predict

#[no_mangle] // Indique au compilateur de ne pas modifier le nom de la fonction pour l'exporter
pub extern "C" fn linear_destroy(model_ptr: *mut LinearModel) { // Détruit la mémoire du modele quand Python a fini
    if !model_ptr.is_null() { // Vérifie que le pointeur n'est pas nul pour éviter les erreurs de segmentation
        unsafe { drop(Box::from_raw(model_ptr)) }; // Libère la mémoire du modèle linéaire (unsafe car on manipule des pointeurs bruts)
    } // fin de la vérification du pointeur nul
} // fin de la fonction linear_destroy