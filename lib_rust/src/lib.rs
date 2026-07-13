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

// sauvegarde : on sort les centres, les poids et gamma vers python

#[no_mangle]
pub extern "C" fn rbf_gamma(ptr: *mut RBFNetwork) -> f64 { unsafe { &*ptr }.gamma() }

#[no_mangle]
pub extern "C" fn rbf_nb_centres(ptr: *mut RBFNetwork) -> usize { unsafe { &*ptr }.nb_centres() }

#[no_mangle]
pub extern "C" fn rbf_taille_centre(ptr: *mut RBFNetwork) -> usize { unsafe { &*ptr }.taille_centre() }

// recopie les centres (mis a plat) dans le tableau fourni par python
#[no_mangle]
pub extern "C" fn rbf_export_centres(ptr: *mut RBFNetwork, out: *mut f64, n: usize) {
    let plat = unsafe { &*ptr }.centres_plat();
    let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n.min(plat.len()) { dst[i] = plat[i]; }
}

// recopie les poids
#[no_mangle]
pub extern "C" fn rbf_export_poids(ptr: *mut RBFNetwork, out: *mut f64, n: usize) {
    let p = unsafe { &*ptr }.poids();
    let dst = unsafe { std::slice::from_raw_parts_mut(out, n) };
    for i in 0..n.min(p.len()) { dst[i] = p[i]; }
}

// recree un modele a partir de parametres deja appris
#[no_mangle]
pub extern "C" fn rbf_charger(centres: *const f64, nb: usize, taille: usize,
                              poids: *const f64, n_poids: usize, gamma: f64) -> *mut RBFNetwork {
    let c = unsafe { std::slice::from_raw_parts(centres, nb * taille) };
    let p = unsafe { std::slice::from_raw_parts(poids, n_poids) };
    Box::into_raw(Box::new(RBFNetwork::depuis_params(c, nb, taille, p, gamma)))
}