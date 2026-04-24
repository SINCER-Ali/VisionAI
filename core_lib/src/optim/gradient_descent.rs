use crate::models::Vector;

#[derive(Clone, Copy, Debug)]
pub struct GradientDescentConfig {
    pub lr: f64,
    pub epochs: usize,
}

impl Default for GradientDescentConfig {
    fn default() -> Self {
        Self {
            lr: 0.001,
            epochs: 5000,
        }
    }
}

pub struct GradientDescent {
    pub learning_rate: f64,
}

impl GradientDescent {
    pub fn new(learning_rate: f64) -> Self {
        GradientDescent { learning_rate }
    }

    pub fn update_weights(&self, weights: &mut Vector, bias: &mut f64, input: &Vector, error: f64) {
        for i in 0..weights.len() {
            weights[i] -= self.learning_rate * 2.0 * error * input[i];
        }
        *bias -= self.learning_rate * 2.0 * error;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_update_weights() {
        let optim = GradientDescent::new(0.1);
        let mut weights = vec![1.0, 1.0];
        let mut bias = 0.0;
        let input = vec![1.0, 1.0];
        let error = 2.0;
        optim.update_weights(&mut weights, &mut bias, &input, error);
        assert!(weights[0] < 1.0, "Le poids doit diminuer");
        assert!(weights[1] < 1.0, "Le poids doit diminuer");
        assert!(bias < 0.0, "Le biais doit diminuer");
    }
}
