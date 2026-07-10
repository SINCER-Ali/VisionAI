// lib ali
// sert a faire le lien entre rust et python

mod rbf; // on dit a rust que rbf.rs existe
use rbf::RBFNetwork;

// cree le modele en memoire et donne son adresse a python
#[no_mangle]
pub extern "C" fn rbf_new(k: usize, gamma: f64) -> *mut RBFNetwork {
    Box::into_raw(Box::new(RBFNetwork::new(k, gamma))) // met le modele sur le tas et retourne l adresse
}

// entraine le modele avec les donnees de python
#[no_mangle]
pub extern "C" fn rbf_train(
    ptr: *mut RBFNetwork,
    data: *const f64,     // toutes les entrees en un seul tableau plat
    n_samples: usize,     // nb d exemples
    n_features: usize,    // taille d un exemple
    targets: *const f64,  // ce qu on attend en sortie
    k: usize,
    iterations: usize,
) {
    let rbf = unsafe { &mut *ptr }; // recupere le modele depuis le pointeur
    let data_slice = unsafe { std::slice::from_raw_parts(data, n_samples * n_features) };
    let data_vec: Vec<Vec<f64>> = data_slice.chunks(n_features).map(|c| c.to_vec()).collect(); // remet les vecteurs dans le bon format
    let targets_slice = unsafe { std::slice::from_raw_parts(targets, n_samples) };
    rbf.train(&data_vec, targets_slice, k, iterations);
}

// retourne une valeur predite pour un point x
#[no_mangle]
pub extern "C" fn rbf_predict(ptr: *mut RBFNetwork, input: *const f64, n_features: usize) -> f64 {
    let rbf = unsafe { &*ptr }; // recupere le modele
    let x = unsafe { std::slice::from_raw_parts(input, n_features) }; // recupere le point x
    rbf.predict(x)
}

// retourne +1 ou -1 pour la classification
#[no_mangle]
pub extern "C" fn rbf_predict_class(ptr: *mut RBFNetwork, input: *const f64, n_features: usize) -> f64 {
    let rbf = unsafe { &*ptr };
    let x = unsafe { std::slice::from_raw_parts(input, n_features) };
    rbf.predict_class(x)
}

// supprime le modele de la memoire quand on a fini
#[no_mangle]
pub extern "C" fn rbf_free(ptr: *mut RBFNetwork) {
    unsafe { drop(Box::from_raw(ptr)); } // libere la memoire
}