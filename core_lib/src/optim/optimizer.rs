// Trait de base pour tous les optimiseurs
// Chaque parametre est identifie par un idx stable
pub trait Optimizer {
    // Met a jour un parametre et retourne la nouvelle valeur
    // idx   : identifiant unique du parametre
    // value : valeur actuelle
    // grad  : gradient de la loss
    fn update(&mut self, idx: usize, value: f64, grad: f64) -> f64;

    // Remet a zero l'etat interne (moments, vitesses)
    fn reset(&mut self);
}
