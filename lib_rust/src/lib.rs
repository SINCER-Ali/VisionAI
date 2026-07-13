// Auteurs : Valentin BROUC (lineaire, SVM) & Nina (MLP)
// Facade "C" : expose les modeles pour les utiliser depuis Python (ctypes).

use std::slice;

// ===================== MODELE LINEAIRE (Valentin BROUC) =====================
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

// ===================== MODELE SVM (Valentin BROUC) =====================
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

// ===================== MODELE MLP / PMC (Nina) =====================
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