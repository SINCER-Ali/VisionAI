// Auteur : Nina
// =====================================================================
//  MLP / PMC (Perceptron Multi-Couches) -- implementation FROM SCRATCH
//  Portage fidele du "NaiveMLP" du cours (slides "Modele Lineaire et PMC").
//  Aucune bibliotheque de ML : uniquement des boucles et des activations.
//
//  Notations (identiques au cours) :
//    d      : tailles des couches, ex. [2, 2, 1].
//    L      : indice de la DERNIERE couche = d.len() - 1.
//    W[l][i][j] : poids du neurone i (couche l-1) vers le neurone j (couche l).
//    X[l][j]    : sortie du neurone j de la couche l. X[l][0] = 1.0 = biais.
//    deltas[l][j] : signal d'erreur du neurone j de la couche l.
// =====================================================================

use rand::Rng; // importe le trait qui donne les methodes aleatoires (gen, gen_range)

// --- Choix de l'activation des couches cachees ---
#[derive(Clone, Copy)] // autorise a copier une Activation (c'est une simple etiquette)
pub enum Activation {   // un type qui ne peut valoir que 3 choses :
    Tanh,               // l'activation tanh
    Sigmoid,            // l'activation sigmoid
    Relu,               // l'activation relu
}                       // fin de l'enum

impl Activation {                                  // bloc des methodes rattachees a Activation
    // Convertit le code recu de Python (0/1/2) en Activation.
    fn from_code(code: usize) -> Activation {      // prend un nombre, renvoie une Activation
        match code {                               // selon la valeur du code :
            1 => Activation::Sigmoid,              // 1 -> sigmoid
            2 => Activation::Relu,                 // 2 -> relu
            _ => Activation::Tanh,                 // tout le reste (0...) -> tanh par defaut
        }                                          // fin du match
    }                                              // fin de from_code

    // f(x) : la fonction d'activation.
    fn apply(self, x: f64) -> f64 {                // applique l'activation choisie a x
        match self {                               // selon l'activation stockee :
            Activation::Tanh => x.tanh(),          // tanh(x), sortie dans [-1, 1]
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()), // sigmoid, sortie dans [0, 1]
            Activation::Relu => if x > 0.0 { x } else { 0.0 }, // relu : x si positif sinon 0
        }                                          // fin du match
    }                                              // fin de apply

    // f'(y) : la derivee, exprimee a partir de la sortie DEJA activee y.
    fn derivative(self, y: f64) -> f64 {           // renvoie la derivee de l'activation
        match self {                               // selon l'activation :
            Activation::Tanh => 1.0 - y * y,       // tanh' = 1 - y^2
            Activation::Sigmoid => y * (1.0 - y),  // sigmoid' = y(1 - y)
            Activation::Relu => if y > 0.0 { 1.0 } else { 0.0 }, // relu' = 1 si actif sinon 0
        }                                          // fin du match
    }                                              // fin de derivative
}                                                  // fin du bloc impl Activation

pub struct MLP {             // la structure qui contient tout le reseau
    d: Vec<usize>,           // tailles des couches, ex. [2, 2, 1]
    l: usize,                // indice de la derniere couche
    w: Vec<Vec<Vec<f64>>>,   // poids  W[l][i][j] (3 dimensions)
    x: Vec<Vec<f64>>,        // sorties X[l][j]
    deltas: Vec<Vec<f64>>,   // erreurs deltas[l][j]
    activation: Activation,  // activation des couches cachees
}                            // fin de la structure

impl MLP {                                         // bloc des methodes du MLP
    // -----------------------------------------------------------------
    //  CONSTRUCTEUR : cree le reseau et initialise les poids au hasard.
    // -----------------------------------------------------------------
    pub fn new(npl: &[usize], activation_code: usize) -> MLP { // recoit l'archi + le code d'activation
        let d: Vec<usize> = npl.to_vec();          // copie les tailles de couches dans d
        let l = npl.len() - 1;                     // indice de la derniere couche = nb couches - 1
        let mut rng = rand::thread_rng();          // prepare le generateur aleatoire

        // --- Poids W ---
        let mut w: Vec<Vec<Vec<f64>>> = Vec::new(); // tableau de poids (vide au depart)
        for layer in 0..=l {                        // pour chaque couche de 0 a L
            w.push(Vec::new());                     // ajoute un espace de poids pour cette couche
            if layer == 0 {                         // la couche 0 (entree)...
                continue;                           // ...n'a pas de poids entrants -> on saute
            }                                       // fin du if
            for _i in 0..=d[layer - 1] {            // pour chaque neurone i de la couche precedente (0 = biais)
                let mut ligne = Vec::new();         // une ligne de poids partant de i
                for j in 0..=d[layer] {             // vers chaque neurone j de la couche courante
                    if j == 0 {                     // colonne j=0 (biais de sortie)...
                        ligne.push(0.0);            // ...inutilisee -> 0
                    } else {                        // sinon...
                        ligne.push(rng.gen::<f64>() * 2.0 - 1.0); // ...poids au hasard dans [-1, 1]
                    }                               // fin du if/else
                }                                   // fin de la boucle j
                w[layer].push(ligne);               // range la ligne de poids dans la couche
            }                                       // fin de la boucle i
        }                                           // fin de la boucle des couches (poids)

        // --- Sorties X et erreurs deltas ---
        let mut x: Vec<Vec<f64>> = Vec::new();      // tableau des sorties (vide)
        let mut deltas: Vec<Vec<f64>> = Vec::new(); // tableau des erreurs (vide)
        for layer in 0..=l {                        // pour chaque couche
            let mut xl = Vec::new();                // les sorties de cette couche
            let mut dl = Vec::new();                // les erreurs de cette couche
            for j in 0..=d[layer] {                 // pour chaque neurone j (0 = biais)
                dl.push(0.0);                       // erreur initialisee a 0
                xl.push(if j == 0 { 1.0 } else { 0.0 }); // sortie : biais a 1, le reste a 0
            }                                       // fin de la boucle j
            x.push(xl);                             // range les sorties de la couche
            deltas.push(dl);                        // range les erreurs de la couche
        }                                           // fin de la boucle des couches (X/deltas)

        MLP {                                       // construit et renvoie le MLP
            d,                                      // les tailles
            l,                                      // l'indice de derniere couche
            w,                                      // les poids
            x,                                      // les sorties
            deltas,                                 // les erreurs
            activation: Activation::from_code(activation_code), // l'activation choisie
        }                                           // fin de la construction
    }                                               // fin de new

    // -----------------------------------------------------------------
    //  PROPAGATION AVANT : on pousse l'entree jusqu'a la sortie.
    // -----------------------------------------------------------------
    fn propagate(&mut self, inputs: &[f64], is_classification: bool) { // propage une entree
        for j in 1..=self.d[0] {                    // pour chaque entree (indice 1.., car 0 = biais)
            self.x[0][j] = inputs[j - 1];           // place la valeur d'entree dans la couche 0
        }                                           // fin du placement de l'entree

        for layer in 1..=self.l {                   // pour chaque couche de 1 a L
            for j in 1..=self.d[layer] {            // pour chaque neurone j de la couche
                let mut total = 0.0;                // accumulateur de la somme ponderee
                for i in 0..=self.d[layer - 1] {     // pour chaque neurone i de la couche precedente (0 = biais)
                    total += self.w[layer][i][j] * self.x[layer - 1][i]; // poids * sortie precedente
                }                                   // fin de la somme
                if layer < self.l {                 // si c'est une couche CACHEE...
                    total = self.activation.apply(total); // ...applique l'activation choisie
                } else if is_classification {       // sinon, si couche de sortie ET classification...
                    total = total.tanh();           // ...applique tanh
                }                                   // (sinon : regression -> on laisse lineaire)
                self.x[layer][j] = total;           // stocke la sortie du neurone j
            }                                       // fin de la boucle j
        }                                           // fin de la boucle des couches
    }                                               // fin de propagate

    // -----------------------------------------------------------------
    //  EXPORT des poids : recopie tous les poids w[l][i][j] a plat dans `out`.
    //  Ordre : couche l (1..L), neurone i (0..d[l-1]), neurone j (0..d[l]).
    //  Sert a SAUVEGARDER le modele (cote Python : get_weights -> JSON).
    // -----------------------------------------------------------------
    pub fn export_weights(&self, out: &mut [f64]) {
        let mut k = 0;                                  // position dans le tableau plat
        for layer in 1..=self.l {                       // pour chaque couche
            for i in 0..=self.d[layer - 1] {            // chaque neurone de depart (0 = biais)
                for j in 0..=self.d[layer] {            // chaque neurone d'arrivee
                    out[k] = self.w[layer][i][j];       // recopie le poids
                    k += 1;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    //  IMPORT des poids : remet les poids a plat de `inp` dans w[l][i][j].
    //  MEME ordre que export_weights. Sert a CHARGER un modele sauvegarde.
    // -----------------------------------------------------------------
    pub fn import_weights(&mut self, inp: &[f64]) {
        let mut k = 0;
        for layer in 1..=self.l {
            for i in 0..=self.d[layer - 1] {
                for j in 0..=self.d[layer] {
                    self.w[layer][i][j] = inp[k];       // remet le poids
                    k += 1;
                }
            }
        }
    }

    // -----------------------------------------------------------------
    //  PREDICTION : propage puis renvoie la couche de sortie.
    // -----------------------------------------------------------------
    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> { // predit une entree
        self.propagate(inputs, is_classification);  // fait la propagation avant
        self.x[self.l][1..].to_vec()                // renvoie la couche de sortie sans le biais (indice 0)
    }                                               // fin de predict

    // -----------------------------------------------------------------
    //  ENTRAINEMENT : descente de gradient stochastique + retropropagation.
    // -----------------------------------------------------------------
    pub fn train(                                   // entraine le reseau
        &mut self,                                  // modifie le reseau (self mutable)
        inputs: &[Vec<f64>],                        // toutes les entrees
        outputs: &[Vec<f64>],                       // toutes les sorties attendues
        steps: usize,                               // nombre d'iterations
        learning_rate: f64,                         // taux d'apprentissage
        is_classification: bool,                    // classification (true) ou regression (false)
    ) {                                             // debut du corps de train
        let mut rng = rand::thread_rng();           // generateur aleatoire

        for _ in 0..steps {                         // on repete `steps` fois :
            // (a) un exemple k au hasard
            let k = rng.gen_range(0..inputs.len()); // tire un indice d'exemple au hasard
            self.propagate(&inputs[k], is_classification); // propage cet exemple

            // (b) erreur de la COUCHE DE SORTIE
            for j in 1..=self.d[self.l] {           // pour chaque neurone de sortie
                self.deltas[self.l][j] = self.x[self.l][j] - outputs[k][j - 1]; // erreur = predit - attendu
                if is_classification {              // en classification...
                    self.deltas[self.l][j] *= 1.0 - self.x[self.l][j].powi(2); // ...* derivee de tanh (1 - X^2)
                }                                   // fin du if
            }                                       // fin de la boucle sortie

            // (c) RETROPROPAGATION vers les couches cachees
            for layer in (2..=self.l).rev() {       // de l'avant-derniere couche en remontant
                for i in 1..=self.d[layer - 1] {    // pour chaque neurone i de la couche precedente
                    let mut total = 0.0;            // accumulateur de l'erreur remontee
                    for j in 1..=self.d[layer] {    // pour chaque neurone j de la couche courante
                        total += self.w[layer][i][j] * self.deltas[layer][j]; // poids * erreur suivante
                    }                               // fin de la somme
                    total *= self.activation.derivative(self.x[layer - 1][i]); // * derivee de l'activation
                    self.deltas[layer - 1][i] = total; // stocke l'erreur du neurone i
                }                                   // fin de la boucle i
            }                                       // fin de la boucle des couches (retropropagation)

            // (d) MISE A JOUR DES POIDS
            for layer in 1..=self.l {               // pour chaque couche
                for i in 0..=self.d[layer - 1] {    // pour chaque neurone i (0 = biais)
                    for j in 1..=self.d[layer] {    // pour chaque neurone j
                        self.w[layer][i][j] -= learning_rate * self.x[layer - 1][i] * self.deltas[layer][j]; // maj : -lr * entree * delta
                    }                               // fin de la boucle j
                }                                   // fin de la boucle i
            }                                       // fin de la boucle des couches (maj des poids)
        }                                           // fin de la boucle des iterations
    }                                               // fin de train
}                                                   // fin du bloc impl MLP
