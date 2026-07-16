// Auteurs : Valentin BROUC (lineaire, SVM) & Thinina (MLP)
// Facade "C" : expose les modeles pour les utiliser depuis Python (ctypes).

use std::slice;

// Modele lineaire (Valentin BROUC)
mod linear;
use linear::LinearModel;

#[no_mangle]
pub extern "C" fn linear_create(input_dim: usize) -> *mut LinearModel {
    Box::into_raw(Box::new(LinearModel::new(input_dim)))
}

#[no_mangle]
pub extern "C" fn linear_train(
    model_ptr: *mut LinearModel,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    input_dim: usize,
    lr: f64,
    epochs: usize,
) {
    let model = unsafe { &mut *model_ptr };
    let all_x = unsafe { slice::from_raw_parts(x_ptr, n_samples * input_dim) };
    let all_y = unsafe { slice::from_raw_parts(y_ptr, n_samples) };
    model.train(all_x, all_y, n_samples, input_dim, lr, epochs);
}

#[no_mangle]
pub extern "C" fn linear_predict(model_ptr: *mut LinearModel, x_ptr: *const f64, input_dim: usize) -> f64 {
    let model = unsafe { &*model_ptr };
    let x = unsafe { slice::from_raw_parts(x_ptr, input_dim) };
    model.predict_class(x)
}

#[no_mangle]
pub extern "C" fn linear_destroy(model_ptr: *mut LinearModel) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr)) };
    }
}

// SVM (Valentin BROUC)
mod svm;
use svm::SVMModel;

#[no_mangle]
pub extern "C" fn svm_create() -> *mut SVMModel {
    Box::into_raw(Box::new(SVMModel::new()))
}

#[no_mangle]
pub extern "C" fn svm_train(
    model_ptr: *mut SVMModel,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    input_dim: usize,
    lr: f64,
    epochs: usize,
    gamma: f64,
) {
    let model = unsafe { &mut *model_ptr };
    let all_x = unsafe { slice::from_raw_parts(x_ptr, n_samples * input_dim) };
    let all_y = unsafe { slice::from_raw_parts(y_ptr, n_samples) };
    model.train(all_x, all_y, n_samples, input_dim, lr, epochs, gamma);
}

#[no_mangle]
pub extern "C" fn svm_predict(model_ptr: *mut SVMModel, x_ptr: *const f64, input_dim: usize) -> f64 {
    let model = unsafe { &*model_ptr };
    let x = unsafe { slice::from_raw_parts(x_ptr, input_dim) };
    model.predict_class(x)
}

#[no_mangle]
pub extern "C" fn svm_destroy(model_ptr: *mut SVMModel) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr)) };
    }
}

// MLP / PMC (Thinina)
mod mlp;
use mlp::MLP;

#[no_mangle]
pub extern "C" fn mlp_create(npl_ptr: *const usize, npl_len: usize, activation_code: usize) -> *mut MLP {
    let npl = unsafe { slice::from_raw_parts(npl_ptr, npl_len) };
    Box::into_raw(Box::new(MLP::new(npl, activation_code)))
}

#[no_mangle]
pub extern "C" fn mlp_train(
    model_ptr: *mut MLP,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    n_inputs: usize,
    n_outputs: usize,
    steps: usize,
    learning_rate: f64,
    is_classification: bool,
) {
    let model = unsafe { &mut *model_ptr };
    let x_flat = unsafe { slice::from_raw_parts(x_ptr, n_samples * n_inputs) };
    let y_flat = unsafe { slice::from_raw_parts(y_ptr, n_samples * n_outputs) };
    let inputs: Vec<Vec<f64>> = x_flat.chunks(n_inputs).map(|c| c.to_vec()).collect();
    let outputs: Vec<Vec<f64>> = y_flat.chunks(n_outputs).map(|c| c.to_vec()).collect();
    model.train(&inputs, &outputs, steps, learning_rate, is_classification);
}

#[no_mangle]
pub extern "C" fn mlp_predict(
    model_ptr: *mut MLP,
    x_ptr: *const f64,
    n_inputs: usize,
    out_ptr: *mut f64,
    n_outputs: usize,
    is_classification: bool,
) {
    let model = unsafe { &mut *model_ptr };
    let inputs = unsafe { slice::from_raw_parts(x_ptr, n_inputs) };
    let pred = model.predict(inputs, is_classification);
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, n_outputs) };
    for i in 0..n_outputs {
        out[i] = pred[i];
    }
}

#[no_mangle]
pub extern "C" fn mlp_export_weights(model_ptr: *mut MLP, out_ptr: *mut f64, len: usize) {
    let model = unsafe { &*model_ptr };
    let out = unsafe { slice::from_raw_parts_mut(out_ptr, len) };
    model.export_weights(out);
}

#[no_mangle]
pub extern "C" fn mlp_import_weights(model_ptr: *mut MLP, in_ptr: *const f64, len: usize) {
    let model = unsafe { &mut *model_ptr };
    let inp = unsafe { slice::from_raw_parts(in_ptr, len) };
    model.import_weights(inp);
}

#[no_mangle]
pub extern "C" fn mlp_destroy(model_ptr: *mut MLP) {
    if !model_ptr.is_null() {
        unsafe { drop(Box::from_raw(model_ptr)) };
    }
}

// RBF (Ali)
mod rbf;
use rbf::RBFNetwork;

#[no_mangle]
pub extern "C" fn rbf_new(k: usize, gamma: f64) -> *mut RBFNetwork {
    Box::into_raw(Box::new(RBFNetwork::new(k, gamma)))
}

#[no_mangle]
pub extern "C" fn rbf_train(
    ptr: *mut RBFNetwork,
    data: *const f64,
    n_samples: usize,
    n_features: usize,
    targets: *const f64,
    k: usize,
    iterations: usize,
) {
    let rbf = unsafe { &mut *ptr };
    let data_slice = unsafe { slice::from_raw_parts(data, n_samples * n_features) };
    let data_vec: Vec<Vec<f64>> = data_slice.chunks(n_features).map(|c| c.to_vec()).collect();
    let targets_slice = unsafe { slice::from_raw_parts(targets, n_samples) };
    rbf.train(&data_vec, targets_slice, k, iterations);
}

#[no_mangle]
pub extern "C" fn rbf_predict(ptr: *mut RBFNetwork, input: *const f64, n_features: usize) -> f64 {
    let rbf = unsafe { &*ptr };
    let x = unsafe { slice::from_raw_parts(input, n_features) };
    rbf.predict(x)
}

#[no_mangle]
pub extern "C" fn rbf_predict_class(ptr: *mut RBFNetwork, input: *const f64, n_features: usize) -> f64 {
    let rbf = unsafe { &*ptr };
    let x = unsafe { slice::from_raw_parts(input, n_features) };
    rbf.predict_class(x)
}

#[no_mangle]
pub extern "C" fn rbf_free(ptr: *mut RBFNetwork) {
    unsafe { drop(Box::from_raw(ptr)); }
}

// sauvegarde : centres, poids et gamma
#[no_mangle]
pub extern "C" fn rbf_gamma(ptr: *mut RBFNetwork) -> f64 { unsafe { &*ptr }.gamma() }

#[no_mangle]
pub extern "C" fn rbf_nb_centres(ptr: *mut RBFNetwork) -> usize { unsafe { &*ptr }.nb_centres() }

#[no_mangle]
pub extern "C" fn rbf_taille_centre(ptr: *mut RBFNetwork) -> usize { unsafe { &*ptr }.taille_centre() }

#[no_mangle]
pub extern "C" fn rbf_export_centres(ptr: *mut RBFNetwork, out: *mut f64, n: usize) {
    let plat = unsafe { &*ptr }.centres_plat();
    let dst = unsafe { slice::from_raw_parts_mut(out, n) };
    for i in 0..n.min(plat.len()) { dst[i] = plat[i]; }
}

#[no_mangle]
pub extern "C" fn rbf_export_poids(ptr: *mut RBFNetwork, out: *mut f64, n: usize) {
    let p = unsafe { &*ptr }.poids();
    let dst = unsafe { slice::from_raw_parts_mut(out, n) };
    for i in 0..n.min(p.len()) { dst[i] = p[i]; }
}

#[no_mangle]
pub extern "C" fn rbf_charger(centres: *const f64, nb: usize, taille: usize,
                              poids: *const f64, n_poids: usize, gamma: f64) -> *mut RBFNetwork {
    let c = unsafe { slice::from_raw_parts(centres, nb * taille) };
    let p = unsafe { slice::from_raw_parts(poids, n_poids) };
    Box::into_raw(Box::new(RBFNetwork::depuis_params(c, nb, taille, p, gamma)))
}

// save/load lineaire (Thinina)
#[no_mangle]
pub extern "C" fn linear_export_weights(ptr: *mut LinearModel, out: *mut f64, len: usize) {
    let w = unsafe { &*ptr }.get_weights();
    let dst = unsafe { slice::from_raw_parts_mut(out, len) };
    for i in 0..len.min(w.len()) { dst[i] = w[i]; }
}

#[no_mangle]
pub extern "C" fn linear_import_weights(ptr: *mut LinearModel, inp: *const f64, len: usize) {
    let w = unsafe { slice::from_raw_parts(inp, len) };
    unsafe { &mut *ptr }.set_weights(w);
}

// valeur brute, pour le un-contre-tous
#[no_mangle]
pub extern "C" fn linear_predict_value(ptr: *mut LinearModel, x_ptr: *const f64, input_dim: usize) -> f64 {
    let model = unsafe { &*ptr };
    let x = unsafe { slice::from_raw_parts(x_ptr, input_dim) };
    model.predict_value(x)
}

// entrainement en regression (Widrow-Hoff)
#[no_mangle]
pub extern "C" fn linear_train_regression(
    model_ptr: *mut LinearModel,
    x_ptr: *const f64,
    y_ptr: *const f64,
    n_samples: usize,
    input_dim: usize,
    lr: f64,
    epochs: usize,
) {
    let model = unsafe { &mut *model_ptr };
    let all_x = unsafe { slice::from_raw_parts(x_ptr, n_samples * input_dim) };
    let all_y = unsafe { slice::from_raw_parts(y_ptr, n_samples) };
    model.train_regression(all_x, all_y, n_samples, input_dim, lr, epochs);
}

// save/load SVM (Thinina)
#[no_mangle]
pub extern "C" fn svm_nb_samples(ptr: *mut SVMModel) -> usize { unsafe { &*ptr }.nb_samples() }
#[no_mangle]
pub extern "C" fn svm_input_dim(ptr: *mut SVMModel) -> usize { unsafe { &*ptr }.input_dim() }
#[no_mangle]
pub extern "C" fn svm_get_bias(ptr: *mut SVMModel) -> f64 { unsafe { &*ptr }.get_bias() }
#[no_mangle]
pub extern "C" fn svm_get_gamma(ptr: *mut SVMModel) -> f64 { unsafe { &*ptr }.get_gamma() }

#[no_mangle]
pub extern "C" fn svm_export_alphas(ptr: *mut SVMModel, out: *mut f64, len: usize) {
    let a = unsafe { &*ptr }.get_alphas();
    let dst = unsafe { slice::from_raw_parts_mut(out, len) };
    for i in 0..len.min(a.len()) { dst[i] = a[i]; }
}

#[no_mangle]
pub extern "C" fn svm_export_y_train(ptr: *mut SVMModel, out: *mut f64, len: usize) {
    let y = unsafe { &*ptr }.get_y_train();
    let dst = unsafe { slice::from_raw_parts_mut(out, len) };
    for i in 0..len.min(y.len()) { dst[i] = y[i]; }
}

#[no_mangle]
pub extern "C" fn svm_export_x_train(ptr: *mut SVMModel, out: *mut f64, len: usize) {
    let dst = unsafe { slice::from_raw_parts_mut(out, len) };
    unsafe { &*ptr }.export_x_train(dst);
}

// valeur de decision continue -> pour le un-contre-tous (argmax)
#[no_mangle]
pub extern "C" fn svm_decision_value(ptr: *mut SVMModel, x_ptr: *const f64, input_dim: usize) -> f64 {
    let model = unsafe { &*ptr };
    let x = unsafe { slice::from_raw_parts(x_ptr, input_dim) };
    model.valeur_decision(x)
}

#[no_mangle]
pub extern "C" fn svm_charger(
    alphas: *const f64, n_alphas: usize,
    bias: f64,
    x_flat: *const f64, x_len: usize,
    y: *const f64,
    n: usize, gamma: f64, dim: usize,
) -> *mut SVMModel {
    let a = unsafe { slice::from_raw_parts(alphas, n_alphas) };
    let xf = unsafe { slice::from_raw_parts(x_flat, x_len) };
    let yy = unsafe { slice::from_raw_parts(y, n) };
    Box::into_raw(Box::new(SVMModel::depuis_params(a, bias, xf, yy, gamma, n, dim)))
}
