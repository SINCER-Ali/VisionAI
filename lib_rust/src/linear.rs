// Auteur: Valentin BROUC
// Modèle linéraire (perceptron de Rosenblatt) pour la classification binaire

use rand::Rng; // Pour générer des nombres aléatoires

pub struct LinearModel {
    weights: Vec<f64>, // Poids du modèle
}

impl LinearModel { // Méthodes du modèle linéaire
    
    pub fn new(input_dim: usize) -> Self { // Constructeur : cree un modele pour des entrees de taille input_dim
        let mut rng = rand::thread_rng(); // Prépare le générateur de nombres aléatoires
        let mut weights = Vec::with_capacity(input_dim + 1); // Crée le vecteur de poids ; +1 pour le biais.
        for _ in 0..(input_dim + 1) { // Pour chaque poids (y compris le biais)
            weights.push(rng.gen_range(-1.0..1.0)); // Initialise les poids aléatoirement entre -1 et 1
        }
        LinearModel { weights } // Retourne le modèle linéaire initialisé avec ses poids
} // fin du new

pub fn predict_value(&self, x: &[f64]) -> f64 { // Prédiction de la valeur (avant activation) pour une entrée x
    let mut sum = self.weights[0]; // On part du biais (équivaut a une entree fictive toujours égale a 1)
    for (i, &xi) in x.iter().enumerate() { // Pour chaque entrée xi
        sum += self.weights[i + 1] * xi; // Ajoute poids x entree ; +1 car weights[0] est le biais
    } // Fin de la boucle sur les entrées (produit scalaire)
    sum // Retourne la somme pondérée (sans le signe d'activation)
} // fin du predict_value

pub fn predict_class(&self, x: &[f64]) -> f64 { // Donne la classe prédite (+1 ou -1)
    if self.predict_value(x) >= 0.0 { // Si la valeur prédite est positive ou nulle
        1.0 // Classe +1
    } else { // Sinon
        -1.0 // Classe -1
    }
}

pub fn train( // Entraine le modèle sur un ensemble d'exemples
    &mut self, // Modifie les poids du modèle
    all_x: &[f64], // Vecteur contenant toutes les entrées (chaque entrée est un vecteur aplati)
    all_y: &[f64], // Vecteur contenant toutes les classes attendues (1 ou -1) pour chaque entrée correspondante
    n_samples: usize, // Nombre d'exemples d'apprentissages
    input_dim: usize, // Taille d'un exemple (nombre d'entrées)
    lr: f64, // Taux d'apprentissage (learning rate) pour la mise à jour des poids
    epochs: usize // Nombre d'itérations sur l'ensemble des exemples pour l'apprentissage
) {
    for _ in 0..epochs { // On répète l'apprentissage epochs fois
        for k in 0..n_samples { // Parcourt chaque exemple k
            let x = &all_x[k * input_dim..(k+1) * input_dim]; // Récupère l'exemple k (sous-vecteur de all_x)}
            let y_true = all_y[k]; // Récupère la classe attendue pour l'exemple k
            let y_pred = self.predict_class(x); // Prédit la classe pour l'exemple k
            let error = y_true - y_pred; // Calcule l'erreur (différence entre la classe attendue et la classe prédite)
            self.weights[0] += lr * error; // Met à jour le biais (poids[0]) en fonction de l'erreur et du taux d'apprentissage
            for i in 0..input_dim { // Pour chaque entrée de l'exemple
                self.weights[i + 1] += lr * error * x[i]; // Met à jour le poids correspondant à l'entrée i
            } // Fin de la boucle sur les poids
    } // Fin de la boucle sur les exemples
} // Fin de la boucle sur les epochs
} // Fin de la méthode train
} // Fin de l'implémentation du modèle linéaire