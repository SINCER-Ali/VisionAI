// Auteur : Valentin BROUC
// SVM avec noyau RBF (kernel trick) : on projette les donnees dans un espace de
// dimension plus grande pour les rendre lineairement separables.

pub struct SVMModel {
    alphas: Vec<f64>,       // un alpha par exemple d'apprentissage
    bias: f64,              // le biais w0
    x_train: Vec<Vec<f64>>, // on conserve les exemples (pour le noyau)
    y_train: Vec<f64>,      // les classes (+1 ou -1) des exemples
    gamma: f64,             // parametre du noyau RBF ; si <= 0 -> noyau lineaire
}

impl SVMModel {
    pub fn new() -> Self {
        SVMModel {
            alphas: Vec::new(),
            bias: 0.0,
            x_train: Vec::new(),
            y_train: Vec::new(),
            gamma: 0.0,       // noyau lineaire par defaut
        }
    }

    // Similarite entre 2 points : RBF si gamma > 0, sinon produit scalaire
    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        if self.gamma > 0.0 {
            let mut dist2 = 0.0;
            for i in 0..a.len() {
                let d = a[i] - b[i];
                dist2 += d * d;
            }
            (-self.gamma * dist2).exp()      // exp(-gamma * ||a-b||^2)
        } else {
            let mut dot = 0.0;
            for i in 0..a.len() {
                dot += a[i] * b[i];
            }
            dot
        }
    }

    // f(x) = bias + somme_n(alpha_n * y_n * K(x_n, x))
    fn decision_value(&self, x: &[f64]) -> f64 {
        let mut sum = self.bias;
        for n in 0..self.x_train.len() {
            sum += self.alphas[n] * self.y_train[n] * self.kernel(&self.x_train[n], x);
        }
        sum
    }

    // Signe de la decision -> classe +1 ou -1
    pub fn predict_class(&self, x: &[f64]) -> f64 {
        if self.decision_value(x) >= 0.0 {
            1.0
        } else {
            -1.0
        }
    }

    // Entrainement : resolution du dual
    pub fn train(
        &mut self,
        all_x: &[f64],
        all_y: &[f64],
        n_samples: usize,
        input_dim: usize,
        lr: f64,
        epochs: usize,
        gamma: f64,
    ) {

        // 1) On conserve les exemples et on initialise les alphas a 0
        self.gamma = gamma;
        self.y_train = all_y.to_vec();
        self.alphas = vec![0.0; n_samples];
        self.x_train = Vec::with_capacity(n_samples);
        for k in 0..n_samples {
            self.x_train
                .push(all_x[k * input_dim..(k + 1) * input_dim].to_vec());
        }

        // 2) Montee de gradient projetee sur le dual : maj des alphas
        for _ in 0..epochs {
            for i in 0..n_samples {
                let mut s = 0.0;
                for j in 0..n_samples {
                    s += self.alphas[j]
                        * self.y_train[j]
                        * self.kernel(&self.x_train[i], &self.x_train[j]);
                }
                self.alphas[i] += lr * (1.0 - self.y_train[i] * s);   // gradient du dual
                if self.alphas[i] < 0.0 {
                    self.alphas[i] = 0.0;
                }
            }
        }

        // 3) Biais (w0) estime a partir des vecteurs de support (alpha > 0)
        let mut somme_bias = 0.0;
        let mut nb_sv = 0;
        for i in 0..n_samples {
            if self.alphas[i] > 1e-6 {
                let mut s = 0.0;
                for j in 0..n_samples {
                    s += self.alphas[j]
                        * self.y_train[j]
                        * self.kernel(&self.x_train[i], &self.x_train[j]);
                }
                somme_bias += self.y_train[i] - s;
                nb_sv += 1;
            }
        }
        if nb_sv > 0 {
            self.bias = somme_bias / nb_sv as f64;   // moyenne : plus stable
        }
    }
}

// Sauvegarde / chargement (Thinina) : expose l'etat du modele.
impl SVMModel {
    pub fn nb_samples(&self) -> usize { self.x_train.len() }
    pub fn input_dim(&self) -> usize {
        if self.x_train.is_empty() { 0 } else { self.x_train[0].len() }
    }
    pub fn get_bias(&self) -> f64 { self.bias }
    pub fn get_gamma(&self) -> f64 { self.gamma }
    pub fn get_alphas(&self) -> &[f64] { &self.alphas }
    pub fn get_y_train(&self) -> &[f64] { &self.y_train }

    // recopie x_train a plat dans out
    pub fn export_x_train(&self, out: &mut [f64]) {
        let mut k = 0;
        for ligne in &self.x_train {
            for &v in ligne {
                if k < out.len() { out[k] = v; k += 1; }
            }
        }
    }

    // valeur continue, pour le un-contre-tous
    pub fn valeur_decision(&self, x: &[f64]) -> f64 { self.decision_value(x) }

    // reconstruit un SVM depuis son etat sauvegarde
    pub fn depuis_params(alphas: &[f64], bias: f64, x_flat: &[f64], y: &[f64], gamma: f64, n: usize, dim: usize) -> SVMModel {
        let mut x_train = Vec::with_capacity(n);
        for k in 0..n {
            x_train.push(x_flat[k * dim..(k + 1) * dim].to_vec());
        }
        SVMModel { alphas: alphas.to_vec(), bias, x_train, y_train: y.to_vec(), gamma }
    }
}
