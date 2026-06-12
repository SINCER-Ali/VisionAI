use crate::math::activations::softmax;
use crate::math::vector::Vector;
use serde::{Deserialize, Serialize};

// Types de noyaux supportes
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum KernelType {
    Linear,
    RBF { gamma: f64 },
    Polynomial { degree: usize, coef0: f64 },
}

impl KernelType {
    pub fn compute(&self, a: &Vector, b: &Vector) -> f64 {
        match self {
            KernelType::Linear => a.dot(b),
            KernelType::RBF { gamma } => {
                let diff = a.sub(b);
                (-gamma * diff.dot(&diff)).exp()
            }
            KernelType::Polynomial { degree, coef0 } => {
                (a.dot(b) + coef0).powi(*degree as i32)
            }
        }
    }
}

// SVM lineaire binaire - hinge loss + SGD (style Pegasos)
// Objectif : (1/2)||w||^2 + C * sum(max(0, 1 - y*(w*x + b)))
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryLinearSVM {
    pub weights: Vector,
    pub bias: f64,
    pub c: f64,
}

impl BinaryLinearSVM {
    fn new(input_size: usize, c: f64) -> Self {
        BinaryLinearSVM { weights: Vector::new(input_size), bias: 0.0, c }
    }

    // labels : +1.0 ou -1.0
    fn train(&mut self, inputs: &[Vector], labels: &[f64], lr: f64, epochs: usize) {
        for epoch in 0..epochs {
            let lr_t = lr / (1.0 + 0.01 * epoch as f64);
            for (x, &y) in inputs.iter().zip(labels.iter()) {
                let margin = y * (self.weights.dot(x) + self.bias);
                if margin < 1.0 {
                    for j in 0..self.weights.len {
                        self.weights.data[j] =
                            (1.0 - lr_t) * self.weights.data[j] + lr_t * self.c * y * x.data[j];
                    }
                    self.bias += lr_t * self.c * y;
                } else {
                    for j in 0..self.weights.len {
                        self.weights.data[j] *= 1.0 - lr_t;
                    }
                }
            }
        }
    }

    fn decision(&self, x: &Vector) -> f64 {
        self.weights.dot(x) + self.bias
    }
}

// SVM a noyau binaire - optimisation duale via SMO simplifie
// Maximise : sum(alpha) - 0.5 * sum_ij(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))
// Sous contraintes : 0 <= alpha_i <= C,  sum(alpha_i * y_i) = 0
#[derive(Clone, Debug, Serialize, Deserialize)]
struct BinaryKernelSVM {
    pub alphas: Vec<f64>,
    pub support_vectors: Vec<Vector>,
    pub sv_labels: Vec<f64>,
    pub bias: f64,
    pub c: f64,
    pub kernel: KernelType,
}

impl BinaryKernelSVM {
    fn new(c: f64, kernel: KernelType) -> Self {
        BinaryKernelSVM {
            alphas: Vec::new(),
            support_vectors: Vec::new(),
            sv_labels: Vec::new(),
            bias: 0.0,
            c,
            kernel,
        }
    }

    fn decision_raw(&self, x: &Vector) -> f64 {
        let mut result = self.bias;
        for i in 0..self.alphas.len() {
            result += self.alphas[i] * self.sv_labels[i]
                * self.kernel.compute(&self.support_vectors[i], x);
        }
        result
    }

    fn decision_from_matrix(
        alphas: &[f64],
        labels: &[f64],
        k: &[Vec<f64>],
        bias: f64,
        i: usize,
    ) -> f64 {
        let sum: f64 = alphas.iter()
            .zip(labels.iter())
            .enumerate()
            .map(|(j, (&a, &y))| a * y * k[j][i])
            .sum();
        sum + bias
    }

    fn train(&mut self, inputs: &[Vector], labels: &[f64], max_iter: usize) {
        let n = inputs.len();
        let eps = 1e-3;
        let tol = 1e-3;

        // Precalcul de la matrice noyau
        let mut k = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in i..n {
                let val = self.kernel.compute(&inputs[i], &inputs[j]);
                k[i][j] = val;
                k[j][i] = val;
            }
        }

        let mut alphas = vec![0.0f64; n];
        let mut bias = 0.0f64;
        let mut iter = 0;
        let mut examine_all = true;

        loop {
            let mut changed = 0;

            let candidates: Vec<usize> = if examine_all {
                (0..n).collect()
            } else {
                (0..n).filter(|&i| alphas[i] > eps && alphas[i] < self.c - eps).collect()
            };

            for &i in &candidates {
                let ei = Self::decision_from_matrix(&alphas, labels, &k, bias, i) - labels[i];
                let ri = labels[i] * ei;

                if (ri < -tol && alphas[i] < self.c) || (ri > tol && alphas[i] > 0.0) {
                    // Choisir j qui maximise |ei - ej|
                    let mut best_j = (i + 1) % n;
                    let mut best_diff = 0.0f64;
                    for j in 0..n {
                        if j == i { continue; }
                        let ej = Self::decision_from_matrix(&alphas, labels, &k, bias, j) - labels[j];
                        let diff = (ei - ej).abs();
                        if diff > best_diff { best_diff = diff; best_j = j; }
                    }

                    let j = best_j;
                    let ej = Self::decision_from_matrix(&alphas, labels, &k, bias, j) - labels[j];
                    let alpha_i_old = alphas[i];
                    let alpha_j_old = alphas[j];

                    let (l, h) = if (labels[i] - labels[j]).abs() < eps {
                        let s = alphas[i] + alphas[j];
                        ((s - self.c).max(0.0), s.min(self.c))
                    } else {
                        let d = alphas[j] - alphas[i];
                        ((-d).max(0.0), (self.c - d).min(self.c))
                    };

                    if (l - h).abs() < eps { continue; }

                    let eta = 2.0 * k[i][j] - k[i][i] - k[j][j];
                    if eta >= 0.0 { continue; }

                    alphas[j] -= labels[j] * (ei - ej) / eta;
                    alphas[j] = alphas[j].max(l).min(h);

                    if (alphas[j] - alpha_j_old).abs() < eps * (alphas[j] + alpha_j_old + eps) {
                        continue;
                    }

                    alphas[i] += labels[i] * labels[j] * (alpha_j_old - alphas[j]);

                    let b1 = bias - ei
                        - labels[i] * (alphas[i] - alpha_i_old) * k[i][i]
                        - labels[j] * (alphas[j] - alpha_j_old) * k[i][j];
                    let b2 = bias - ej
                        - labels[i] * (alphas[i] - alpha_i_old) * k[i][j]
                        - labels[j] * (alphas[j] - alpha_j_old) * k[j][j];

                    bias = if alphas[i] > eps && alphas[i] < self.c - eps { b1 }
                           else if alphas[j] > eps && alphas[j] < self.c - eps { b2 }
                           else { (b1 + b2) / 2.0 };

                    changed += 1;
                }
            }

            iter += 1;
            if iter >= max_iter { break; }
            if examine_all { examine_all = false; }
            else if changed == 0 { examine_all = true; }
        }

        self.bias = bias;
        self.alphas.clear();
        self.support_vectors.clear();
        self.sv_labels.clear();
        for i in 0..n {
            if alphas[i] > eps {
                self.alphas.push(alphas[i]);
                self.support_vectors.push(inputs[i].clone());
                self.sv_labels.push(labels[i]);
            }
        }
    }
}

// SVM multi-classes via One-vs-Rest
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SVM {
    pub c: f64,
    pub kernel: KernelType,
    pub n_classes: usize,
    linear_classifiers: Vec<BinaryLinearSVM>,
    kernel_classifiers: Vec<BinaryKernelSVM>,
    use_kernel: bool,
}

impl SVM {
    pub fn new_linear(c: f64) -> Self {
        SVM {
            c,
            kernel: KernelType::Linear,
            n_classes: 0,
            linear_classifiers: Vec::new(),
            kernel_classifiers: Vec::new(),
            use_kernel: false,
        }
    }

    pub fn new_kernel(c: f64, kernel: KernelType) -> Self {
        SVM {
            c,
            kernel: kernel.clone(),
            n_classes: 0,
            linear_classifiers: Vec::new(),
            kernel_classifiers: Vec::new(),
            use_kernel: true,
        }
    }

    // Retourne softmax des scores OvR, argmax = classe predite
    pub fn predict(&self, input: &Vector) -> Vector {
        assert!(self.n_classes > 0, "SVM non entraine");
        let scores: Vec<f64> = if self.use_kernel {
            self.kernel_classifiers.iter().map(|clf| clf.decision_raw(input)).collect()
        } else {
            self.linear_classifiers.iter().map(|clf| clf.decision(input)).collect()
        };
        softmax(&Vector::from_vec(scores))
    }

    // Entrainement OvR : un classifieur binaire par classe
    // lr     : taux d'apprentissage (SVM lineaire uniquement)
    // epochs : iterations (lineaire) ou max_iter SMO (noyau)
    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], lr: f64, epochs: usize) {
        assert!(!inputs.is_empty());
        let n_classes = targets[0].len;
        self.n_classes = n_classes;
        let input_size = inputs[0].len;

        if self.use_kernel {
            self.kernel_classifiers = (0..n_classes).map(|cls| {
                let labels: Vec<f64> = targets.iter()
                    .map(|t| if t.argmax() == cls { 1.0 } else { -1.0 })
                    .collect();
                let mut clf = BinaryKernelSVM::new(self.c, self.kernel.clone());
                clf.train(inputs, &labels, epochs.max(50));
                clf
            }).collect();
        } else {
            self.linear_classifiers = (0..n_classes).map(|cls| {
                let labels: Vec<f64> = targets.iter()
                    .map(|t| if t.argmax() == cls { 1.0 } else { -1.0 })
                    .collect();
                let mut clf = BinaryLinearSVM::new(input_size, self.c);
                clf.train(inputs, &labels, lr, epochs);
                clf
            }).collect();
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

    fn and_data() -> (Vec<Vector>, Vec<Vector>) {
        (
            vec![
                Vector::from_vec(vec![0.0, 0.0]),
                Vector::from_vec(vec![0.0, 1.0]),
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![1.0, 1.0]),
            ],
            vec![
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![1.0, 0.0]),
                Vector::from_vec(vec![0.0, 1.0]),
            ],
        )
    }

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
    fn linear_svm_output_shape() {
        let (inputs, targets) = and_data();
        let mut svm = SVM::new_linear(1.0);
        svm.train(&inputs, &targets, 0.1, 500);
        assert_eq!(svm.predict(&inputs[0]).len, 2);
    }

    #[test]
    fn linear_svm_softmax_sums_to_one() {
        let (inputs, targets) = and_data();
        let mut svm = SVM::new_linear(1.0);
        svm.train(&inputs, &targets, 0.1, 500);
        for x in &inputs {
            let sum: f64 = svm.predict(x).data.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn linear_svm_and_converges() {
        let (inputs, targets) = and_data();
        let expected = vec![0usize, 0, 0, 1];
        let mut ok = false;
        for _ in 0..5 {
            let mut svm = SVM::new_linear(10.0);
            svm.train(&inputs, &targets, 0.05, 2000);
            if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
                ok = true; break;
            }
        }
        assert!(ok, "SVM lineaire doit converger sur AND");
    }

    #[test]
    fn linear_svm_or_converges() {
        let inputs = vec![
            Vector::from_vec(vec![0.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![1.0, 0.0]),
            Vector::from_vec(vec![1.0, 1.0]),
        ];
        let targets = vec![
            Vector::from_vec(vec![1.0, 0.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![0.0, 1.0]),
            Vector::from_vec(vec![0.0, 1.0]),
        ];
        let expected = vec![0usize, 1, 1, 1];
        let mut ok = false;
        for _ in 0..5 {
            let mut svm = SVM::new_linear(10.0);
            svm.train(&inputs, &targets, 0.05, 2000);
            if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
                ok = true; break;
            }
        }
        assert!(ok, "SVM lineaire doit converger sur OR");
    }

    #[test]
    fn kernel_svm_xor_rbf() {
        let (inputs, targets) = xor_data();
        let expected = vec![0usize, 1, 1, 0];
        let mut ok = false;
        for _ in 0..5 {
            let mut svm = SVM::new_kernel(5.0, KernelType::RBF { gamma: 1.0 });
            svm.train(&inputs, &targets, 0.0, 200);
            if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
                ok = true; break;
            }
        }
        assert!(ok, "SVM noyau RBF doit converger sur XOR");
    }

    #[test]
    fn svm_json_roundtrip() {
        let (inputs, targets) = and_data();
        let mut svm = SVM::new_linear(1.0);
        svm.train(&inputs, &targets, 0.1, 500);
        svm.save_json("__svm_test.json").unwrap();
        let loaded = SVM::load_json("__svm_test.json").unwrap();
        let out_orig = svm.predict(&inputs[0]);
        let out_load = loaded.predict(&inputs[0]);
        for (a, b) in out_orig.data.iter().zip(out_load.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        std::fs::remove_file("__svm_test.json").ok();
    }

    #[test]
    fn svm_binary_roundtrip() {
        let (inputs, targets) = and_data();
        let mut svm = SVM::new_linear(1.0);
        svm.train(&inputs, &targets, 0.1, 500);
        svm.save_binary("__svm_test.bin").unwrap();
        let loaded = SVM::load_binary("__svm_test.bin").unwrap();
        let out_orig = svm.predict(&inputs[0]);
        let out_load = loaded.predict(&inputs[0]);
        for (a, b) in out_orig.data.iter().zip(out_load.data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        std::fs::remove_file("__svm_test.bin").ok();
    }
}
