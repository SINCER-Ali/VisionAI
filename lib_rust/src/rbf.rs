//  RBF ALI
// Permet de prédire en placant des zones d'influence.


// Note les centres, les poids et le gamma dans le modéle
pub struct RBFNetwork {
    centers: Vec<Vec<f64>>,
    weights: Vec<f64>,
    gamma: f64,
}


// Constructeur qui crée un modèle vide
impl RBFNetwork {
    pub fn new(_k: usize, gamma: f64) -> Self {
        RBFNetwork {
            centers: Vec::new(),
            weights: Vec::new(),
            gamma,
        }
    }


    // Formulaire Gaussienne qui permet de calculer la distence entre un point et un centre avec la formule du cours
    // Si le résultat est proche du centre alors il est proche de 1 (formule:exp(-γ × ||x - centre||²))
    fn gaussian(&self, x: &[f64], center: &[f64]) -> f64 {
        let dist_sq: f64 = x.iter() //iter parcours la liste
            .zip(center.iter()) // relie les deux liste ensemble
            .map(|(xi, ci)| (xi - ci).powi(2)) // pour chaque paire faire le calcul
            .sum(); // addition
        (-self.gamma * dist_sq).exp() //mis au carré avec l'exponentiel comme dans la formule
    }

    // K-means, permet de trouver les centre, les assigner et se déplacer vers la moyenne de ses points
    fn kmeans(&mut self, data: &[Vec<f64>], k: usize, iterations: usize) {
        let n = data.len(); // nombre d'exemples
        let dim = data[0].len(); // taille d'un exemple (ex: 32x32 = 1024)

        // Étape 1 : on choisit K exemples au hasard comme centres de départ
        self.centers = (0..k).map(|i| data[i * n / k].clone()).collect();

        for _ in 0..iterations { // on répète "iterations" fois
            // Étape 2 : assigner chaque point à son centre le plus proche
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); k];

            for (i, point) in data.iter().enumerate() {
                // trouver le centre le plus proche de ce point
                let nearest = self.centers.iter().enumerate()
                    .map(|(j, c)| {
                        let dist: f64 = point.iter().zip(c.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum();
                        (j, dist)
                    })
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap().0;

                clusters[nearest].push(i); // ce point appartient au cluster "nearest"
            }

            // Étape 3 : déplacer chaque centre à la moyenne de son cluster
            for (j, cluster) in clusters.iter().enumerate() {
                if cluster.is_empty() { continue; } // si un cluster est vide, on skip
                let mut new_center = vec![0.0; dim];
                for &idx in cluster {
                    for d in 0..dim {
                        new_center[d] += data[idx][d]; // additionne tous les points
                    }
                }
                for d in 0..dim {
                    self.centers[j][d] = new_center[d] / cluster.len() as f64; // moyenne
                }
            }
        }
    }

    // Construit la matrice Phi : phi[i][j] = distance gaussienne entre exemple i et centre j
        fn build_phi(&self, data: &[Vec<f64>]) -> Vec<Vec<f64>> {
            data.iter().map(|point| {                          // pour chaque exemple
                self.centers.iter().map(|center| {             // pour chaque centre
                    self.gaussian(point, center)               // calcule la gaussienne
                }).collect()
                //  une ligne de phi
            }).collect()                                       //  toute la matrice
        }


    // calcule les poids W pour que le modele soit bon
    fn solve_weights(phi: &[Vec<f64>], targets: &[f64]) -> Vec<f64> {
        let k = phi[0].len(); // nb de centres


        // PtP = Phi transposé fois Phi, ca donne une matrice K×K
        let mut ptp = vec![vec![0.0; k]; k];
        for i in 0..k {
            for j in 0..k {
                for row in phi {
                    ptp[i][j] += row[i] * row[j]; // produit matriciel
                }
            }
        }

        // PtY = Phi transposé fois Y, ca donne un vecteur de taille K
        let mut pty = vec![0.0; k];
        for i in 0..k {
            for (row, &y) in phi.iter().zip(targets.iter()) {
                pty[i] += row[i] * y; // on multiplie chaque ligne par sa cible
            }
        }

        // on colle PtY a droite de PtP pour faire la matrice augmentée
        let mut aug: Vec<Vec<f64>> = ptp.iter().enumerate()
            .map(|(i, row)| {
                let mut r = row.clone();
                r.push(pty[i]); // PtY va a droite
                r
            }).collect();

        // Gauss-Jordan : on transforme la gauche en identite, la droite devient W
        for col in 0..k {
            let inv = 1.0 / aug[col][col];       // inverse du pivot
            for v in aug[col].iter_mut() { *v *= inv; } // on normalise la ligne
            for row in 0..k {
                if row == col { continue; }       // on skip la ligne pivot
                let factor = aug[row][col];
                for j in 0..=k {
                    let val = aug[col][j] * factor;
                    aug[row][j] -= val;           // on elimine la colonne
                }
            }
        }

        // W est dans la derniere colonne, cest notre reponse
        aug.iter().map(|row| row[k]).collect()
    }
    // lance les 3 etapes pour entrainer le modele
    pub fn train(&mut self, data: &[Vec<f64>], targets: &[f64], k: usize, iterations: usize) {
        self.kmeans(data, k, iterations);          // trouve les centres
        let phi = self.build_phi(data);            // construit phi
        self.weights = Self::solve_weights(&phi, targets); // calcule les poids W
    }

    // predit un nombre pour un point x
    pub fn predict(&self, x: &[f64]) -> f64 {
        self.weights.iter()
            .zip(self.centers.iter())
            .map(|(w, center)| w * self.gaussian(x, center)) // poids fois gaussienne pour chaque centre
            .sum() // on additionne tout
    }

    // predit la classe +1 ou -1
    pub fn predict_class(&self, x: &[f64]) -> f64 {
        if self.predict(x) >= 0.0 { 1.0 } else { -1.0 } // si positif cest +1 sinon -1
    }

    // pour sauvegarder / recharger le modele
    pub fn gamma(&self) -> f64 { self.gamma }
    pub fn nb_centres(&self) -> usize { self.centers.len() }
    pub fn taille_centre(&self) -> usize {
        if self.centers.is_empty() { 0 } else { self.centers[0].len() }
    }
    pub fn centres_plat(&self) -> Vec<f64> {
        self.centers.iter().flatten().copied().collect() // centres mis bout a bout
    }
    pub fn poids(&self) -> Vec<f64> { self.weights.clone() }

    // reconstruit un modele deja appris (sans relancer l'entrainement)
    pub fn depuis_params(centres: &[f64], nb: usize, taille: usize, poids: &[f64], gamma: f64) -> Self {
        let centers = centres.chunks(taille).map(|c| c.to_vec()).take(nb).collect();
        RBFNetwork { centers, weights: poids.to_vec(), gamma }
    }

}