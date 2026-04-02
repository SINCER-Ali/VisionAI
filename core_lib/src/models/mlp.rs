use crate::math::activations::{Activation, softmax};
use crate::math::vector::Vector;
use crate::optim::gradient_descent::GradientDescentConfig;
use rand::Rng;
use serde::{Deserialize, Serialize};

fn default_activation() -> Activation {
    Activation::ReLU
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layer {
    pub weights: Vec<Vector>,
    pub biases: Vector,
    pub input_size: usize,
    pub output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = (0..output_size)
            .map(|_| {
                let data: Vec<f64> = (0..input_size)
                    .map(|_| rng.gen_range(-0.5..0.5))
                    .collect();
                Vector::from_vec(data)
            })
            .collect();

        let biases = Vector::new(output_size);

        Layer { weights, biases, input_size, output_size }
    }

    pub fn forward(&self, input: &Vector) -> Vector {
        let mut output = self.biases.clone();
        for i in 0..self.output_size {
            output.data[i] += self.weights[i].dot(input);
        }
        output
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MLP {
    pub layers: Vec<Layer>,
    #[serde(default = "default_activation")]
    pub hidden_activation: Activation,
}

impl MLP {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        MLP {
            layers,
            hidden_activation: Activation::ReLU,
        }
    }

    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.hidden_activation = activation;
        self
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(&current);
            current = if i == self.layers.len() - 1 {
                softmax(&z)
            } else {
                self.hidden_activation.apply(&z)
            };
        }
        current
    }

    pub fn train(&mut self, inputs: &[Vector], targets: &[Vector], cfg: GradientDescentConfig) {
        for _epoch in 0..cfg.epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let mut zs = Vec::new();
                let mut activations = vec![input.clone()];

                for (i, layer) in self.layers.iter().enumerate() {
                    let z = layer.forward(activations.last().unwrap());
                    zs.push(z.clone());
                    let a = if i == self.layers.len() - 1 {
                        softmax(&z)
                    } else {
                        self.hidden_activation.apply(&z)
                    };
                    activations.push(a);
                }

                let output = activations.last().unwrap();
                let mut delta = output.sub(target);

                for l in (0..self.layers.len()).rev() {
                    let a_prev = &activations[l];

                    for i in 0..self.layers[l].output_size {
                        for j in 0..self.layers[l].input_size {
                            self.layers[l].weights[i].data[j] -= cfg.lr * delta.data[i] * a_prev.data[j];
                        }
                        self.layers[l].biases.data[i] -= cfg.lr * delta.data[i];
                    }

                    if l > 0 {
                        let mut new_delta = Vector::new(self.layers[l].input_size);
                        for j in 0..self.layers[l].input_size {
                            for i in 0..self.layers[l].output_size {
                                new_delta.data[j] += self.layers[l].weights[i].data[j] * delta.data[i];
                            }
                        }
                        let rd = self.hidden_activation.derivative(&zs[l - 1]);
                        delta = new_delta.hadamard(&rd);
                    }
                }
            }
        }
    }

    /// Sauvegarde le modele au format JSON
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Charge un modele depuis un fichier JSON
    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let model: MLP = serde_json::from_str(&content)?;
        Ok(model)
    }

    /// Sauvegarde le modele au format binaire (bincode)
    pub fn save_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    /// Charge un modele depuis un fichier binaire
    pub fn load_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let model: MLP = bincode::deserialize(&data)?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict_output_size() {
        let mlp = MLP::new(&[4, 8, 3]);
        let input = Vector::from_vec(vec![0.5, 0.2, 0.8, 0.1]);
        let output = mlp.predict(&input);
        assert_eq!(output.len, 3);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mlp = MLP::new(&[4, 8, 3]);
        let input = Vector::from_vec(vec![0.5, 0.2, 0.8, 0.1]);
        let output = mlp.predict(&input);
        let sum: f64 = output.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Softmax doit sommer a 1.0, got {}", sum);
    }
}
