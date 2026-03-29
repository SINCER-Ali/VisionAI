use crate::math::vector::Vector;
use crate::optim::gradient_descent::GradientDescentConfig;
use rand::Rng;

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

fn relu(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| x.max(0.0)).collect();
    Vector::from_vec(data)
}

fn relu_derivative(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| if *x > 0.0 { 1.0 } else { 0.0 }).collect();
    Vector::from_vec(data)
}

fn softmax(v: &Vector) -> Vector {
    let max = v.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.data.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Vector::from_vec(exps.iter().map(|x| x / sum).collect())
}

pub struct MLP {
    pub layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let layers = layer_sizes
            .windows(2)
            .map(|w| Layer::new(w[0], w[1]))
            .collect();
        MLP { layers }
    }

    pub fn predict(&self, input: &Vector) -> Vector {
        let mut current = input.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let z = layer.forward(&current);
            current = if i == self.layers.len() - 1 {
                softmax(&z)
            } else {
                relu(&z)
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
                        relu(&z)
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
                        let rd = relu_derivative(&zs[l - 1]);
                        delta = new_delta.hadamard(&rd);
                    }
                }
            }
        }
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