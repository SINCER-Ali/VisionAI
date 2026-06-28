use crate::math::activations::softmax;
use crate::math::vector::Vector;
use crate::optim::gradient_descent::GradientDescentConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};

// RBF Network (Reseau a Fonctions de Base Radiale)
// Couche cachee : n_centers neurones, activation phi(x) = exp(-gamma * ||x - c||^2)
// Couche sortie : combinaison lineaire + softmax
// Entrainement  : centres aleatoires, poids par moindres carres (Gauss-Jordan), raffinement gradient
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RBF {
    pub centers: Vec<Vector>,
    pub weights: Vec<Vector>,
    pub biases: Vector,
    pub gamma: f64,
    pub n_centers: usize,
    pub n_outputs: usize,
    pub lambda: f64,
}

impl RBF {
    pub fn new(n_centers: usize, gamma: f64, n_outputs: usize) -> Self {
        RBF {
            centers: Vec::new(),
            weights: Vec::new(),
            biases: Vector::new(n_outputs),
            gamma,
            n_centers,
            n_outputs,
            lambda: 1e-4,
        }
    }

    pub fn with_lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    // Noyau gaussien : exp(-gamma * ||x - c||^2)
    pub fn rbf_kernel(x: &Vector, center: &Vector, gamma: f64) -> f64 {
        let diff = x.sub(center);
        let sq_dist = diff.dot(&diff);
        (-gamma * sq_dist).exp()
    }

    // Activations de la couche cachee pour une entree x
    pub fn compute_activations(&self, x: &Vector) -> Vector {
        let data: Vec<f64> = self
            .centers
            .iter()
            .map(|c| Self::rbf_kernel(x, c, self.gamma))
            .collect();
        Vector::from_vec(data)
    }

    // Initialisation des centres par echantillonnage aleatoire (Fisher-Yates)
    fn init_centers_random(inputs: &[Vector], n_centers: usize) -> Vec<Vector> {
        let n = inputs.len();
        let k = n_centers.min(n);
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
        indices[..k]
            .iter()
            .map(|&idx| inputs[idx].clone())
            .collect()
    }

    // Resout W = (Phi^T Phi + lambda I)^-1 * Phi^T Y via Gauss-Jordan avec pivot partiel
    fn solve_weights(
        phi: &[Vec<f64>],
        targets: &[Vector],
        n_centers: usize,
        n_outputs: usize,
        lambda: f64,
    ) -> (Vec<Vector>, Vector) {
        // Phi^T Phi
        let mut ptp = vec![vec![0.0f64; n_centers]; n_centers];
        for phi_row in phi {
            for i in 0..n_centers {
                for j in 0..n_centers {
                    ptp[i][j] += phi_row[i] * phi_row[j];
                }
            }
        }
        for (i, row) in ptp.iter_mut().enumerate() {
            row[i] += lambda;
        }

        // Phi^T Y
        let mut pty = vec![vec![0.0f64; n_outputs]; n_centers];
        for (phi_row, target) in phi.iter().zip(targets.iter()) {
            for (phi_val, pty_row) in phi_row.iter().zip(pty.iter_mut()) {
                for (pty_val, td) in pty_row.iter_mut().zip(target.data.iter()) {
                    *pty_val += phi_val * td;
                }
            }
        }

        // Matrice augmentee [A | B]
        let w = n_centers + n_outputs;
        let mut aug: Vec<Vec<f64>> = (0..n_centers)
            .map(|i| {
                let mut row = vec![0.0f64; w];
                row[..n_centers].copy_from_slice(&ptp[i]);
                row[n_centers..].copy_from_slice(&pty[i]);
                row
            })
            .collect();

        // Elimination de Gauss-Jordan
        for col in 0..n_centers {
            let (mut max_row, mut max_val) = (col, aug[col][col].abs());
            for (row_idx, aug_row) in aug.iter().enumerate().skip(col + 1) {
                if aug_row[col].abs() > max_val {
                    max_val = aug_row[col].abs();
                    max_row = row_idx;
                }
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            if pivot.abs() < 1e-14 {
                continue;
            }
            let inv = 1.0 / pivot;
            for v in aug[col].iter_mut() {
                *v *= inv;
            }
            let pivot_row = aug[col].clone();
            for (row, aug_row) in aug.iter_mut().enumerate() {
                if row == col {
                    continue;
                }
                let f = aug_row[col];
                if f.abs() < 1e-14 {
                    continue;
                }
                for (v, &p) in aug_row.iter_mut().zip(pivot_row.iter()) {
                    *v -= f * p;
                }
            }
        }

        let weights: Vec<Vector> = (0..n_outputs)
            .map(|j| Vector::from_vec((0..n_centers).map(|i| aug[i][n_centers + j]).collect()))
            .collect();
        (weights, Vector::new(n_outputs))
    }

    // Sortie brute avant softmax (utile pour la regression)
    pub fn predict_raw(&self, input: &Vector) -> Vector {
        assert!(!self.centers.is_empty(), "RBF non entraine");
        let act = self.compute_activations(input);
        let mut output = self.biases.clone();
        for (j, w) in self.weights.iter().enumerate() {
            output.data[j] += w.dot(&act);
        }
        output
    }

    // Prediction avec softmax pour la classification
    pub fn predict(&self, input: &Vector) -> Vector {
        softmax(&self.predict_raw(input))
    }

    // Entrainement :
    //   1. centres aleatoires
    //   2. poids par moindres carres regularises
    //   3. cfg.epochs etapes de raffinement gradient avec taux cfg.lr
    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], cfg: GradientDescentConfig) {
        assert!(!inputs.is_empty(), "Donnees vides");
        assert_eq!(inputs.len(), targets.len());
        let n_outputs = targets[0].len;
        self.n_outputs = n_outputs;
        self.biases = Vector::new(n_outputs);
        self.centers = Self::init_centers_random(inputs, self.n_centers);
        self.n_centers = self.centers.len();
        let phi: Vec<Vec<f64>> = inputs
            .iter()
            .map(|x| {
                self.centers
                    .iter()
                    .map(|c| Self::rbf_kernel(x, c, self.gamma))
                    .collect()
            })
            .collect();
        let (weights, biases) =
            Self::solve_weights(&phi, targets, self.n_centers, n_outputs, self.lambda);
        self.weights = weights;
        self.biases = biases;
        if cfg.epochs > 0 && cfg.lr > 0.0 {
            for _ in 0..cfg.epochs {
                for (x, target) in inputs.iter().zip(targets.iter()) {
                    let pred = self.predict(x);
                    let act = self.compute_activations(x);
                    let delta = pred.sub(target);
                    for (j, w) in self.weights.iter_mut().enumerate() {
                        for k in 0..self.n_centers {
                            w.data[k] -= cfg.lr * delta.data[j] * act.data[k];
                        }
                        self.biases.data[j] -= cfg.lr * delta.data[j];
                    }
                }
            }
        }
    }

    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&content)?)
    }
    pub fn save_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }
    pub fn load_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        Ok(bincode::deserialize(&data)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_data() -> (Vec<Vector>, Vec<Vector>) {
        (
            vec![
                Vector::from_vec(vec![0.0, 0.0]),
                Vector::from_vec(vec![0.0, 1.0]),
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![1.0, 1.0]),
            ],
            vec![
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![0.0, 1.0]),
                Vector::from_vec(vec![0.0, 1.0]),
                Vector::from_vec(vec![1.0, 0.0]),
            ],
        )
    }

    #[test]
    fn rbf_output_shape() {
        let (inputs, targets) = xor_data();
        let mut rbf = RBF::new(4, 1.0, 2);
        rbf.train(
            &inputs,
            &targets,
            GradientDescentConfig { lr: 0.0, epochs: 0 },
        );
        assert_eq!(rbf.predict(&inputs[0]).len, 2);
    }

    #[test]
    fn rbf_softmax_sums_to_one() {
        let (inputs, targets) = xor_data();
        let mut rbf = RBF::new(4, 1.0, 2);
        rbf.train(
            &inputs,
            &targets,
            GradientDescentConfig { lr: 0.0, epochs: 0 },
        );
        for x in &inputs {
            let sum: f64 = rbf.predict(x).data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn rbf_xor_converges() {
        let (inputs, targets) = xor_data();
        let expected = vec![0usize, 1, 1, 0];
        let mut ok = false;
        for _ in 0..10 {
            let mut rbf = RBF::new(4, 2.0, 2).with_lambda(1e-8);
            rbf.train(
                &inputs,
                &targets,
                GradientDescentConfig {
                    lr: 0.05,
                    epochs: 300,
                },
            );
            if inputs
                .iter()
                .zip(expected.iter())
                .all(|(x, &e)| rbf.predict(x).argmax() == e)
            {
                ok = true;
                break;
            }
        }
        assert!(ok, "RBF doit converger sur XOR");
    }

    #[test]
    fn rbf_json_roundtrip() {
        let (inputs, targets) = xor_data();
        let mut rbf = RBF::new(4, 1.0, 2);
        rbf.train(
            &inputs,
            &targets,
            GradientDescentConfig { lr: 0.0, epochs: 0 },
        );
        rbf.save_json("__rbf_test.json").unwrap();
        let loaded = RBF::load_json("__rbf_test.json").unwrap();
        let out_orig = rbf.predict(&inputs[0]);
        let out_load = loaded.predict(&inputs[0]);
        for (a, b) in out_orig.data.iter().zip(out_load.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        std::fs::remove_file("__rbf_test.json").ok();
    }

    #[test]
    fn rbf_binary_roundtrip() {
        let (inputs, targets) = xor_data();
        let mut rbf = RBF::new(4, 1.0, 2);
        rbf.train(
            &inputs,
            &targets,
            GradientDescentConfig { lr: 0.0, epochs: 0 },
        );
        rbf.save_binary("__rbf_test.bin").unwrap();
        let loaded = RBF::load_binary("__rbf_test.bin").unwrap();
        let out_orig = rbf.predict(&inputs[0]);
        let out_load = loaded.predict(&inputs[0]);
        for (a, b) in out_orig.data.iter().zip(out_load.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        std::fs::remove_file("__rbf_test.bin").ok();
    }
}
