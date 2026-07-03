// Auteur : Valentin BROUC
// On projette les données dans un espace de dimension plus grande pour les rendre linéairement séparables (SVM avec noyau RBF - kernel trick)

pub struct SVMModel {
    // Structure du modèle
    alphas: Vec<f64>,       // un alpha par exemple d'apprentissage
    bias: f64,              // le biais w0
    x_train: Vec<Vec<f64>>, // on conserve les exemples (pour le noyau)
    y_train: Vec<f64>,      // les classes (+1 ou -1) des exemples
    gamma: f64,             // paramètre du noyau RBF ; si <= 0 -> noyau linéaire
}

impl SVMModel {
    // Début du bloc des méthodes rattachées au modèle
    pub fn new() -> Self {
        SVMModel {
            alphas: Vec::new(),  // initialise le vecteur des alphas à vide
            bias: 0.0,           // initialise le biais à 0
            x_train: Vec::new(), // initialise le vecteur des exemples à vide
            y_train: Vec::new(), // initialise le vecteur des classes à vide
            gamma: 0.0,          // initialise le paramètre du noyau à 0 (noyau linéaire par défaut)
        }
    }

    // Noyau K(a, b) : produit scalaire (linéraire) ou RBF selon gamma
    fn kernel(&self, a: &[f64], b: &[f64]) -> f64 {
        // on calcule la similarité entre deux points a et b
        if self.gamma > 0.0 {
            // si gamma > 0 : on utilise le noyau RBF
            let mut dist2 = 0.0; // distance au carré entre a et b
            for i in 0..a.len() {
                // pour chaque dimension
                let d = a[i] - b[i]; // différence entre les coordonnées
                dist2 += d * d; // on ajoute le carré de la différence à la distance
            }
            (-self.gamma * dist2).exp() // noyau RBF : exp(-gamma * ||a-b||^2)
        } else {
            // sinon on utilise le noyau linéaire
            let mut dot = 0.0; // produit scalaire entre a et b
            for i in 0..a.len() {
                // pour chaque dimension
                dot += a[i] * b[i]; // on ajoute le produit des coordonnées à la somme
            } // fin de la boucle sur les dimensions
            dot // noyau linéaire : produit scalaire
        } // fin du if sur gamma
    } // fin de la fonction kernel

    // f(x) = bias + somme_n(alpha_n * y_n * K(x_n, x))
    fn decision_value(&self, x: &[f64]) -> f64 {
        // calcule la valeur brute de décision pour un point x
        let mut sum = self.bias; // on part du biais
        for n in 0..self.x_train.len() {
            // pour chaque exemple d'apprentissage
            sum += self.alphas[n] * self.y_train[n] * self.kernel(&self.x_train[n], x);
            // vote pondéré de l'exemple n
        } // fin de la boucle sur les exemples
        sum // retourne la somme (valeur continue, sans le signe)
    } // fin de décision_value

    pub fn predict_class(&self, x: &[f64]) -> f64 {
        // donne la classe prédite d'un point x
        if self.decision_value(x) >= 0.0 {
            1.0
        } else {
            -1.0
        } // signe de la décision -> classe +1 ou -1
    } // fin de predict_class

    pub fn train(
        // entraine le modèle (résout le dual)
        &mut self,        // on modifie le modèle
        all_x: &[f64], // vecteur contenant toutes les entrées (chaque entrée est un vecteur aplati)
        all_y: &[f64], // vecteur contenant toutes les classes attendues (1 ou -1) pour chaque entrée correspondante
        n_samples: usize, // nombre d'exemples d'apprentissage
        input_dim: usize, // taille d'un exemple (nombre d'entrées)
        lr: f64,       // taux d'apprentissage (learning rate) pour la mise à jour des alphas
        epochs: usize, // nombre d'itérations sur l'ensemble des exemples pour l'apprentissage
        gamma: f64,    // paramètre du noyau RBF ; si <= 0 -> noyau linéaire sinon RBF
    ) {
        // début du corps de train

        // 1) On conserve les exemples et on initialise les alphas à 0
        self.gamma = gamma; // on stocke le paramètre du noyau
        self.y_train = all_y.to_vec(); // on copie les classes dans le modèle
        self.alphas = vec![0.0; n_samples]; // un alpha par exemple, initialisés à 0
        self.x_train = Vec::with_capacity(n_samples); // on réserve de la place pour les exemples
        for k in 0..n_samples {
            // pour chaque exemple k
            self.x_train
                .push(all_x[k * input_dim..(k + 1) * input_dim].to_vec()); // on découpe l'exemple k et on le stocke dans x_train
        } // fin de la boucle de découpage des exemples

        // 2) Montée de gradient projetée sur le dual (on met à jour les alphas)
        for _ in 0..epochs {
            // on répète l'apprentissage epochs fois
            for i in 0..n_samples {
                // pour chaque exemple i
                let mut s = 0.0; // somme des contributions des autres exemples (accumulateur : somme_j alpha_j * y_j * K(x_j, x_i))
                for j in 0..n_samples {
                    // pour chaque exemple j
                    s += self.alphas[j]
                        * self.y_train[j]
                        * self.kernel(&self.x_train[i], &self.x_train[j]); // on ajoute la contribution de l'exemple j à la somme
                } // fin de la boucle interne sur les exemples j
                self.alphas[i] += lr * (1.0 - self.y_train[i] * s); // mise à jour de l'alpha_i selon la règle de gradient projeté (gradient du dual = 1 - y_i * s)
                if self.alphas[i] < 0.0 {
                    self.alphas[i] = 0.0; // si l'alpha est devenu négatif
                    self.alphas[i] = 0.0; // on le ramène à 0 (projection sur le domaine des alphas >= 0)
                } // fin du if de projection
            } // fin de la boucle externe sur les exemples i
        } // fin de la boucle sur les epochs

        // 3) On calcule le biais (w0) à partir des vecteurs de support (alpha > 0)
        let mut somme_bias = 0.0; // accumulateur pour la somme des biais estimés
        let mut nb_sv = 0; // compteur de vecteurs de support
        for i in 0..n_samples {
            // pour chaque exemple i
            if self.alphas[i] > 1e-6 {
                // si alpha_i est non nul -> c'est un vecteur de support
                let mut s = 0.0; // accumulateur : somme_j alpha_j * y_j * K(x_j, x_i)
                for j in 0..n_samples {
                    // pour chaque exemple j
                    s += self.alphas[j]
                        * self.y_train[j]
                        * self.kernel(&self.x_train[i], &self.x_train[j]); // on ajoute la contribution de l'exemple j à la somme
                } // fin de la boucle interne sur les exemples j
                somme_bias += self.y_train[i] - s; // w0 pour ce vecteur de support = y_i - s (car 1/y_i= y_i)
                nb_sv += 1; // on a trouvé un vecteur de support
            } // fin du if sur alpha_i
        } // fin de la boucle sur les exemples i
        if nb_sv > 0 {
            // si on a trouvé au moins un vecteur de support
            self.bias = somme_bias / nb_sv as f64; // biais final = moyenne des estimations (plus stable)
        } // fin du if
    } // fin de la méthode train
} // fin du bloc implémenté
