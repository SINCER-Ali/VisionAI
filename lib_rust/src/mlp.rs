// Auteur : Thinina
// MLP / PMC (Perceptron Multi-Couches)
// Notations : d = tailles des couches, L = derniere couche,
// W[l][i][j] = poids du neurone i (couche l-1) vers j (couche l),
// X[l][j] = sortie du neurone j, X[l][0] = 1.0 = biais.

use rand::Rng;

#[derive(Clone, Copy)]
pub enum Activation {
    Tanh,
    Sigmoid,
    Relu,
}

impl Activation {
    fn from_code(code: usize) -> Activation {
        match code {
            1 => Activation::Sigmoid,
            2 => Activation::Relu,
            _ => Activation::Tanh,
        }
    }

    fn apply(self, x: f64) -> f64 {
        match self {
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Relu => if x > 0.0 { x } else { 0.0 },
        }
    }

    // derivees exprimees a partir de la sortie DEJA activee
    fn derivative(self, y: f64) -> f64 {
        match self {
            Activation::Tanh => 1.0 - y * y,
            Activation::Sigmoid => y * (1.0 - y),
            Activation::Relu => if y > 0.0 { 1.0 } else { 0.0 },
        }
    }
}

pub struct MLP {
    d: Vec<usize>,           // tailles des couches
    l: usize,                // indice de la derniere couche
    w: Vec<Vec<Vec<f64>>>,   // poids w[couche][i][j]
    x: Vec<Vec<f64>>,        // sorties x[couche][neurone]
    deltas: Vec<Vec<f64>>,   // signal d'erreur
    activation: Activation,  // activation des couches cachees
}

impl MLP {
    pub fn new(npl: &[usize], activation_code: usize) -> MLP {
        let d: Vec<usize> = npl.to_vec();
        let l = npl.len() - 1;
        let mut rng = rand::thread_rng();

        // poids : la couche 0 n'a pas de poids entrants
        let mut w: Vec<Vec<Vec<f64>>> = Vec::new();
        for layer in 0..=l {
            w.push(Vec::new());
            if layer == 0 {
                continue;
            }
            for _i in 0..=d[layer - 1] {
                let mut ligne = Vec::new();
                for j in 0..=d[layer] {
                    if j == 0 {
                        ligne.push(0.0);   // colonne du biais : non utilisee
                    } else {
                        ligne.push(rng.gen::<f64>() * 2.0 - 1.0);
                    }
                }
                w[layer].push(ligne);
            }
        }

        // sorties et erreurs ; l'indice 0 de chaque couche est le biais
        let mut x: Vec<Vec<f64>> = Vec::new();
        let mut deltas: Vec<Vec<f64>> = Vec::new();
        for layer in 0..=l {
            let mut xl = Vec::new();
            let mut dl = Vec::new();
            for j in 0..=d[layer] {
                dl.push(0.0);
                xl.push(if j == 0 { 1.0 } else { 0.0 });
            }
            x.push(xl);
            deltas.push(dl);
        }

        MLP {
            d,
            l,
            w,
            x,
            deltas,
            activation: Activation::from_code(activation_code),
        }
    }

    // Propagation avant : de l'entree jusqu'a la sortie
    fn propagate(&mut self, inputs: &[f64], is_classification: bool) {
        for j in 1..=self.d[0] {
            self.x[0][j] = inputs[j - 1];
        }

        for layer in 1..=self.l {
            for j in 1..=self.d[layer] {
                let mut total = 0.0;
                for i in 0..=self.d[layer - 1] {
                    total += self.w[layer][i][j] * self.x[layer - 1][i];
                }
                if layer < self.l {
                    total = self.activation.apply(total);   // couche cachee
                } else if is_classification {
                    total = total.tanh();                   // sortie : tanh, ou lineaire en regression
                }
                self.x[layer][j] = total;
            }
        }
    }

    // Poids a plat, ordre : couche l, neurone i, neurone j
    pub fn export_weights(&self, out: &mut [f64]) {
        let mut k = 0;
        for layer in 1..=self.l {
            for i in 0..=self.d[layer - 1] {
                for j in 0..=self.d[layer] {
                    out[k] = self.w[layer][i][j];
                    k += 1;
                }
            }
        }
    }

    // Meme ordre que export_weights
    pub fn import_weights(&mut self, inp: &[f64]) {
        let mut k = 0;
        for layer in 1..=self.l {
            for i in 0..=self.d[layer - 1] {
                for j in 0..=self.d[layer] {
                    self.w[layer][i][j] = inp[k];
                    k += 1;
                }
            }
        }
    }

    pub fn predict(&mut self, inputs: &[f64], is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.x[self.l][1..].to_vec()   // sans le biais
    }

    // Descente de gradient stochastique + retropropagation
    pub fn train(
        &mut self,
        inputs: &[Vec<f64>],
        outputs: &[Vec<f64>],
        steps: usize,
        learning_rate: f64,
        is_classification: bool,
    ) {
        let mut rng = rand::thread_rng();

        for _ in 0..steps {
            // (a) un exemple au hasard
            let k = rng.gen_range(0..inputs.len());
            self.propagate(&inputs[k], is_classification);

            // (b) erreur de la couche de sortie
            for j in 1..=self.d[self.l] {
                self.deltas[self.l][j] = self.x[self.l][j] - outputs[k][j - 1];
                if is_classification {
                    self.deltas[self.l][j] *= 1.0 - self.x[self.l][j].powi(2);
                }
            }

            // (c) retropropagation vers les couches cachees
            for layer in (2..=self.l).rev() {
                for i in 1..=self.d[layer - 1] {
                    let mut total = 0.0;
                    for j in 1..=self.d[layer] {
                        total += self.w[layer][i][j] * self.deltas[layer][j];
                    }
                    total *= self.activation.derivative(self.x[layer - 1][i]);
                    self.deltas[layer - 1][i] = total;
                }
            }

            // (d) mise a jour des poids
            for layer in 1..=self.l {
                for i in 0..=self.d[layer - 1] {
                    for j in 1..=self.d[layer] {
                        self.w[layer][i][j] -= learning_rate * self.x[layer - 1][i] * self.deltas[layer][j];
                    }
                }
            }
        }
    }
}
