//  RBF ALI
// Permet de prédire en placant des zones d'influence.

pub struct RBFNetwork {
    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
}

impl RBFNetwork {
    pub fn new(_k: usize, gamma: f64) -> Self {
        RBFNetwork {
            centers: Vec::new(),
            weights: Vec::new(),
            gamma,
        }
    }

    // Gaussienne du cours : exp(-gamma * ||x - centre||^2)
    // Proche du centre -> proche de 1
    fn gaussian(&self, x: &[f64], center: &[f64]) -> f64 {
        let dist_sq: f64 = x.iter()
            .zip(center.iter())
            .map(|(xi, ci)| (xi - ci).powi(2))
            .sum();
        (-self.gamma * dist_sq).exp()
    }

    // K-means : trouve les k centres
    fn kmeans(&mut self, data: &[Vec<f64>], k: usize, iterations: usize) {
        let n = data.len();
        let dim = data[0].len();

        // Étape 1 : k centres de depart, pris a intervalle regulier (un exemple tous les n/k).
        // Deterministe, donc reproductible.
        self.centers = (0..k).map(|i| data[i * n / k].clone()).collect();

        for _ in 0..iterations {
            // Étape 2 : assigner chaque point a son centre le plus proche
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];

            for (i, point) in data.iter().enumerate() {
                let nearest = self.centers.iter().enumerate()
                    .map(|(j, c)| {
                        let dist: f64 = point.iter().zip(c.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (j, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap().0;

                clusters[nearest].push(i);
            }

            // Étape 3 : deplacer chaque centre a la moyenne de son cluster
            for (j, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() { continue; }
                let mut new_center = vec![0.0; dim];
                for &idx in cluster {
                    for d in 0..dim {
                        new_center[d] += data[idx][d];
                    }
                }
                for d in 0..dim {
                    self.centers[j][d] = new_center[d] / cluster.len() as f64;
                }
            }
        }
    }

    // phi[i][j] = gaussienne entre l'exemple i et le centre j
        fn build_phi(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|point| {
                self.centers.iter().map(|center| {
                    self.gaussian(point, center)
                }).collect()
            }).collect()
        }


    // Moindres carres : resout (Phi^T Phi) W = Phi^T Y par Gauss-Jordan
    fn solve_weights(phi: &[Vec<f64>], targets: &[f64]) -> Vec<f64> {
        let k = phi[0].len();

        // PtP = Phi transpose fois Phi -> matrice K x K
        let mut ptp = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for row in phi {
                    ptp[i][j] += row[i] * row[j];
                }
            }
        }

        // PtY = Phi transpose fois Y -> vecteur de taille K
        let mut pty = vec![0.0; k];
        for i in 0..k {
            for (row, &y) in phi.iter().zip(targets.iter()) {
                pty[i] += row[i] * y;
            }
        }

        // matrice augmentee : PtY colle a droite de PtP
        let mut aug: Vec<Vec<f64>> = ptp.iter().enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.push(pty[i]);
                r
            }).collect();

        // Gauss-Jordan : la gauche devient l'identite, la droite devient W
        for col in 0..k {
            let inv = 1.0 / aug[col][col];
            for v in aug[col].iter_mut() { *v *= inv; }
            for row in 0..k {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..=k {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;
                }
            }
        }

        aug.iter().map(|row| row[k]).collect()   // W = derniere colonne
    }

    // Entrainement en 3 etapes
    pub fn train(&mut self, data: &[Vec<f64>], targets: &[f64], k: usize, iterations: usize) {
        self.kmeans(data, k, iterations);
        let phi = self.build_phi(data);
        self.weights = Self::solve_weights(&phi, targets);
    }

    // Somme ponderee des gaussiennes -> valeur continue
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.weights.iter()
            .zip(self.centers.iter())
            .map(|(w, center)| w * self.gaussian(x, center))
            .sum()
    }

    // Signe de predict -> classe +1 ou -1
    pub fn predict_class(&self, x: &[f64]) -> f64 {
        if self.predict(x) >= 0.0 { 1.0 } else { -1.0 }
    }

    // pour sauvegarder / recharger le modele
    pub fn gamma(&self) -> f64 { self.gamma }
    pub fn nb_centres(&self) -> usize { self.centers.len() }
    pub fn taille_centre(&self) -> usize {
        if self.centers.is_empty() { 0 } else { self.centers[0].len() }
    }
    pub fn centres_plat(&self) -> Vec<f64> {
        self.centers.iter().flatten().copied().collect()
    }
    pub fn poids(&self) -> Vec<f64> { self.weights.clone() }

    // reconstruit un modele deja appris (sans relancer l'entrainement)
    pub fn depuis_params(centres: &[f64], nb: usize, taille: usize, poids: &[f64], gamma: f64) -> Self {
        let centers = centres.chunks(taille).map(|c| c.to_vec()).take(nb).collect();
        RBFNetwork { centers, weights: poids.to_vec(), gamma }
    }

}
