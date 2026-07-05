// Auteur : Nina
// Facade "C" : expose le MLP pour l'utiliser depuis Python (ctypes).
//
// NOTE : on ne declare ici que `mod mlp;`. Les autres modeles
// (linear, svm, rbf) sont geres par les coequipiers sur leurs branches ;
// on ne les compile pas ici pour rester concentre sur le MLP.

use std::slice; // pour reconstruire des slices a partir de pointeurs bruts

mod mlp; // Importe le module mlp.rs
use mlp::MLP; // Utilise la structure MLP du module mlp.rs

// ---------------------------------------------------------------------
//  mlp_create : cree un MLP a partir des tailles de couches (ex. [2,2,1]).
//  Renvoie un pointeur ("ticket") que Python garde et repasse ensuite.
// ---------------------------------------------------------------------
#[no_mangle]
pub extern "C" fn mlp_create(
    npl_ptr: *const usize,
    npl_len: usize,
    activation_code: usize, // 0 = tanh, 1 = sigmoid, 2 = relu
) -> *mut MLP {
    let npl = unsafe { slice::from_raw_parts(npl_ptr, npl_len) };
    Box::into_raw(Box::new(MLP::new(npl, activation_code)))
}

// ---------------------------------------------------------------------
//  mlp_train : entraine le reseau.
//  x_ptr / y_ptr sont les donnees APLATIES (lues ligne par ligne).
// ---------------------------------------------------------------------
#[no_mangle]
pub extern "C" fn mlp_train(
    model_ptr: *mut MLP,
    x_ptr: *const f64,        // toutes les entrées aplaties
    y_ptr: *const f64,        // toutes les sorties attendues aplaties
    n_samples: usize,         // nombre d'exemples
    n_inputs: usize,          // taille d'une entrée
    n_outputs: usize,         // taille d'une sortie
    steps: usize,             // nombre d'itérations (tirages aléatoires)
    learning_rate: f64,
    is_classification: bool,  // true = classification (tanh en sortie), false = régression
) {
    let model = unsafe { &mut *model_ptr };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_inputs) };
    let y_flat = unsafe { slice::from_raw_parts(y_ptr, n_samples * n_outputs) };
    // On recoupe les tableaux plats en lignes.
    let inputs: Vec<Vec<f64>> = x_flat.chunks(n_inputs).map(|c| c.to_vec()).collect();
    let outputs: Vec<Vec<f64>> = y_flat.chunks(n_outputs).map(|c| c.to_vec()).collect();
    model.train(&inputs, &outputs, steps, learning_rate, is_classification);
}

// ---------------------------------------------------------------------
//  mlp_predict : predit UN exemple. Le resultat est ecrit dans out_ptr
//  (tampon de n_outputs cases alloue cote Python), car le MLP peut avoir
//  plusieurs sorties (multi-classe / regression).
// ---------------------------------------------------------------------
#[no_mangle]
pub extern "C" fn mlp_predict(
    model_ptr: *mut MLP,
    x_ptr: *const f64,        // l'entrée
    n_inputs: usize,          // taille de l'entrée
    out_ptr: *mut f64,        // tampon de sortie (alloué côté Python)
    n_outputs: usize,         // taille de la sortie
    is_classification: bool,
) {
    let model = unsafe { &mut *model_ptr };
    let inputs = unsafe { slice::from_raw_parts(x_ptr, n_inputs) };
    let pred = model.predict(inputs, is_classification);
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, n_outputs) };
    for i in 0..n_outputs {
        out[i] = pred[i]; // on recopie la prédiction dans le tampon Python
    }
}

// ---------------------------------------------------------------------
//  mlp_destroy : libere la memoire du reseau (appele depuis Python a la fin).
// ---------------------------------------------------------------------
#[no_mangle]
pub extern "C" fn mlp_destroy(model_ptr: *mut MLP) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr)) };
    }
}
