// Auteur: Valentin BROUC
// Modèle linéraire (perceptron de Rosenblatt) pour la classification binaire

use rand::Rng;

pub struct LinearModel {
    weights: Vec<f64>,   // weights[0] = biais
}

impl LinearModel {

    pub fn new(input_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = Vec::with_capacity(input_dim + 1);   // +1 pour le biais
        for _ in 0..(input_dim + 1) {
            weights.push(rng.gen_range(-1.0..1.0));
        }
        LinearModel { weights }
    }

    // Somme ponderee, avant activation
    pub fn predict_value(&self, x: &[f64]) -> f64 {
        let mut sum = self.weights[0];
        for (i, &xi) in x.iter().enumerate() {
            sum += self.weights[i + 1] * xi;
        }
        sum
    }

    // Classe predite : +1 ou -1
    pub fn predict_class(&self, x: &[f64]) -> f64 {
        if self.predict_value(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    // Regle du perceptron : maj des poids selon l'erreur, pour chaque exemple
    pub fn train(
        &mut self,
        all_x: &[f64],
        all_y: &[f64],
        n_samples: usize,
        input_dim: usize,
        lr: f64,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for k in 0..n_samples {
                let x = &all_x[k * input_dim..(k + 1) * input_dim];
                let y_true = all_y[k];
                let y_pred = self.predict_class(x);
                let error = y_true - y_pred;
                self.weights[0] += lr * error;
                for i in 0..input_dim {
                    self.weights[i + 1] += lr * error * x[i];
                }
            }
        }
    }
}

// Sauvegarde / chargement (Thinina)
impl LinearModel {
    pub fn get_weights(&self) -> &[f64] { &self.weights }
    pub fn set_weights(&mut self, w: &[f64]) { self.weights = w.to_vec(); }
}

// Regression (Thinina) : regle de Widrow-Hoff.
// Le train ci-dessus est le perceptron (il apprend sur le SIGNE) -> classification.
// Ici l'erreur se calcule sur la VALEUR CONTINUE (predict_value) -> regression.
impl LinearModel {
    pub fn train_regression(
        &mut self,
        all_x: &[f64],
        all_y: &[f64],
        n_samples: usize,
        input_dim: usize,
        lr: f64,
        epochs: usize,
    ) {
        for _ in 0..epochs {
            for k in 0..n_samples {
                let x = &all_x[k * input_dim..(k + 1) * input_dim];
                let y_pred = self.predict_value(x);   // valeur continue, pas le signe
                let error = all_y[k] - y_pred;
                self.weights[0] += lr * error;
                for i in 0..input_dim {
                    self.weights[i + 1] += lr * error * x[i];
                }
            }
        }
    }
}
