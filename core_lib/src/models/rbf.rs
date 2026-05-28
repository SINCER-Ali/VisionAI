use crate::math::activations::softmax;
use crate::math::matrix::Matrix;
use crate::math::vector::Vector;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// Config d'entraînement — miroir de GradientDescentConfig (MLP)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct RBFConfig {
    pub lr: f64,
    pub epochs: usize,
    pub n_centers: usize,
    pub sigma: f64,
    /// true  → régression (sortie brute)
    /// false → classification (softmax sur la sortie)
    pub regression: bool,
}

impl Default for RBFConfig {
    fn default() -> Self {
        RBFConfig {
            lr: 0.01,
            epochs: 500,
            n_centers: 10,
            sigma: 1.0,
            regression: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────
// Noyau gaussien : φ(x, c) = exp(−‖x − c‖² / 2σ²)
// ─────────────────────────────────────────────────────────────

fn gaussian_rbf(x: &Vector, center: &Vector, sigma: f64) -> f64 {
    let diff = x.sub(center);
    let dist_sq = diff.dot(&diff);
    (-dist_sq / (2.0 * sigma * sigma)).exp()
}

// ─────────────────────────────────────────────────────────────
// Struct principale
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RBFNetwork {
    /// Centres des neurones cachés
    pub centers: Vec<Vector>,
    /// Largeur des gaussiennes
    pub sigma: f64,
    /// Matrice de poids couche cachée → sortie
    /// shape : output_size × (n_centers + 1)  — le +1 absorbe le biais
    pub weights: Matrix,
    pub input_size: usize,
    pub output_size: usize,
    pub n_centers: usize,
}

impl RBFNetwork {
    // ── Constructeur ─────────────────────────────────────────

    pub fn new(input_size: usize, output_size: usize, n_centers: usize, sigma: f64) -> Self {
        RBFNetwork {
            centers: Vec::new(),
            sigma,
            weights: Matrix::new(output_size, n_centers + 1),
            input_size,
            output_size,
            n_centers,
        }
    }

    // ── Initialisation des centres ────────────────────────────

    /// Échantillonnage aléatoire parmi les données (Fisher-Yates partiel)
    pub fn init_centers_random(&mut self, data: &[Vector]) {
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        for i in 0..self.n_centers.min(data.len()) {
            let j = rng.gen_range(i..data.len());
            indices.swap(i, j);
        }
        self.centers = indices
            .iter()
            .take(self.n_centers)
            .map(|&i| data[i].clone())
            .collect();

        // Heuristique σ : d_max / √(2k)
        self.sigma = self.auto_sigma();
        // Réinitialise les poids à zéro après avoir fixé les centres
        self.weights = Matrix::new(self.output_size, self.n_centers + 1);
    }

    fn auto_sigma(&self) -> f64 {
        if self.centers.len() < 2 {
            return 1.0;
        }
        let mut d_max: f64 = 0.0;
        for i in 0..self.centers.len() {
            for j in (i + 1)..self.centers.len() {
                let diff = self.centers[i].sub(&self.centers[j]);
                let d = diff.dot(&diff).sqrt();
                if d > d_max {
                    d_max = d;
                }
            }
        }
        d_max / (2.0 * self.centers.len() as f64).sqrt()
    }

    // ── Couche cachée ─────────────────────────────────────────

    /// Vecteur d'activations de la couche cachée.
    /// Taille : n_centers + 1  (data[0] = 1.0 = biais)
    fn hidden_activations(&self, x: &Vector) -> Vector {
        let mut data = vec![1.0]; // biais
        for center in &self.centers {
            data.push(gaussian_rbf(x, center, self.sigma));
        }
        Vector::from_vec(data)
    }

    // ── Forward ───────────────────────────────────────────────

    /// Sortie brute (pré-softmax), taille output_size
    fn forward_raw(&self, x: &Vector) -> Vector {
        let phi = self.hidden_activations(x);
        let cols = self.n_centers + 1;
        let mut out_data = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = 0.0;
            for j in 0..cols {
                sum += self.weights.data[i * cols + j] * phi.data[j];
            }
            out_data[i] = sum;
        }
        Vector::from_vec(out_data)
    }

    // ── Predict ───────────────────────────────────────────────
    // Même signature que MLP::predict : prend &Vector, retourne Vector

    pub fn predict(&self, input: &Vector) -> Vector {
        softmax(&self.forward_raw(input))
    }

    /// Variante régression (sortie brute sans softmax)
    pub fn predict_regression(&self, input: &Vector) -> Vector {
        self.forward_raw(input)
    }

    // ── Train ─────────────────────────────────────────────────
    // Même signature que MLP::train

    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], cfg: &RBFConfig) {
        assert!(
            !self.centers.is_empty(),
            "Appelez init_centers_random() avant train()"
        );

        let cols = self.n_centers + 1;

        for epoch in 0..cfg.epochs {
            let mut total_loss = 0.0;

            for (x, target) in inputs.iter().zip(targets.iter()) {
                let phi = self.hidden_activations(x);
                let raw = self.forward_raw(x);

                // delta = prédiction − cible
                let (delta, loss) = if cfg.regression {
                    let d = raw.sub(target);
                    let l = d.data.iter().map(|e| e * e).sum::<f64>() / d.len as f64;
                    (d, l)
                } else {
                    let proba = softmax(&raw);
                    let d = proba.sub(target);
                    let l = d.data.iter().map(|e| e * e).sum::<f64>() / d.len as f64;
                    (d, l)
                };

                total_loss += loss;

                // W[i][j] -= lr * delta[i] * phi[j]
                for i in 0..self.output_size {
                    for j in 0..cols {
                        self.weights.data[i * cols + j] -=
                            cfg.lr * delta.data[i] * phi.data[j];
                    }
                }
            }

            if epoch % 50 == 0 {
                println!(
                    "[RBF] Époque {:>4} — Loss: {:.6}",
                    epoch,
                    total_loss / inputs.len() as f64
                );
            }
        }
    }

    // ── Métriques ─────────────────────────────────────────────

    pub fn accuracy(&self, inputs: &[Vector], targets: &[Vector]) -> f64 {
        let correct = inputs
            .iter()
            .zip(targets.iter())
            .filter(|(x, t)| self.predict(x).argmax() == t.argmax())
            .count();
        correct as f64 / inputs.len() as f64
    }

    pub fn mse(&self, inputs: &[Vector], targets: &[Vector]) -> f64 {
        let total: f64 = inputs
            .iter()
            .zip(targets.iter())
            .map(|(x, t)| {
                let diff = self.predict_regression(x).sub(t);
                diff.data.iter().map(|e| e * e).sum::<f64>()
            })
            .sum();
        total / (inputs.len() as f64 * self.output_size as f64)
    }

    // ── Persistance — même pattern que LinearModel et MLP ─────

    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let model: RBFNetwork = serde_json::from_str(&content)?;
        Ok(model)
    }

    pub fn save_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let model: RBFNetwork = bincode::deserialize(&data)?;
        Ok(model)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn xor_dataset() -> (Vec<Vector>, Vec<Vector>) {
        let inputs = vec![
            Vector::from_vec(vec![0.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![1.0, 0.0]),
            Vector::from_vec(vec![1.0, 1.0]),
        ];
        // One-hot : classe 0 = [1,0], classe 1 = [0,1]
        let targets = vec![
            Vector::from_vec(vec![1.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![1.0, 0.0]),
        ];
        (inputs, targets)
    }

    #[test]
    fn test_hidden_activations_size() {
        let (inputs, _) = xor_dataset();
        let mut rbf = RBFNetwork::new(2, 2, 4, 1.0);
        rbf.init_centers_random(&inputs);
        let phi = rbf.hidden_activations(&inputs[0]);
        assert_eq!(phi.len, 5, "biais + n_centers = 5");
    }

    #[test]
    fn test_predict_output_size() {
        let (inputs, _) = xor_dataset();
        let mut rbf = RBFNetwork::new(2, 2, 4, 1.0);
        rbf.init_centers_random(&inputs);
        let out = rbf.predict(&inputs[0]);
        assert_eq!(out.len, 2);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let (inputs, _) = xor_dataset();
        let mut rbf = RBFNetwork::new(2, 2, 4, 1.0);
        rbf.init_centers_random(&inputs);
        let out = rbf.predict(&inputs[0]);
        let sum: f64 = out.data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax doit sommer à 1.0, got {sum}"
        );
    }

    #[test]
    fn test_train_xor_loss_diminue() {
        let (inputs, targets) = xor_dataset();
        let mut rbf = RBFNetwork::new(2, 2, 4, 1.0);
        rbf.init_centers_random(&inputs);

        // Loss initiale : on mesure l'accuracy avant (doit être mauvaise)
        let acc_before = rbf.accuracy(&inputs, &targets);

        let cfg = RBFConfig {
            lr: 0.1,
            epochs: 500,
            n_centers: 4,
            sigma: rbf.sigma,
            regression: false,
        };
        rbf.train(&inputs, &targets, &cfg);

        let acc_after = rbf.accuracy(&inputs, &targets);
        println!("Accuracy avant: {acc_before:.3} | après: {acc_after:.3}");
        assert!(
            acc_after >= acc_before,
            "L'accuracy doit augmenter après entraînement : {acc_before:.3} → {acc_after:.3}"
        );
    }

    #[test]
    fn test_save_load_json() {
        let (inputs, targets) = xor_dataset();
        let mut rbf = RBFNetwork::new(2, 2, 4, 1.0);
        rbf.init_centers_random(&inputs);
        rbf.train(&inputs, &targets, &RBFConfig::default());

        let path = std::env::temp_dir()
            .join("rbf_test.json")
            .to_string_lossy()
            .to_string();

        rbf.save_json(&path).unwrap();
        let loaded = RBFNetwork::load_json(&path).unwrap();

        // Comparaison avec epsilon au lieu de == (précision f64 JSON)
        assert_eq!(rbf.weights.data.len(), loaded.weights.data.len());
        for (a, b) in rbf.weights.data.iter().zip(loaded.weights.data.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "Poids divergent après reload : {a} ≠ {b}"
            );
        }
        assert_eq!(rbf.centers.len(), loaded.centers.len());
    }

    #[test]
    fn test_regression_y_eq_x1_plus_x2() {
        let inputs = vec![
            Vector::from_vec(vec![0.0, 0.0]),
            Vector::from_vec(vec![1.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Vector::from_vec(vec![0.0]),
            Vector::from_vec(vec![1.0]),
            Vector::from_vec(vec![1.0]),
            Vector::from_vec(vec![2.0]),
        ];
        let mut rbf = RBFNetwork::new(2, 1, 4, 1.0);
        rbf.init_centers_random(&inputs);
        let cfg = RBFConfig {
            lr: 0.05,
            epochs: 1000,
            n_centers: 4,
            sigma: rbf.sigma,
            regression: true,
        };
        rbf.train(&inputs, &targets, &cfg);
        let mse = rbf.mse(&inputs, &targets);
        assert!(mse < 0.1, "MSE doit être < 0.1, got {mse}");
    }
}
