use super::{Model, TrainConfig, Vector};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearModel {
    pub weights: Vector,
    pub bias: f64,
    pub input_size: usize,
}

impl LinearModel {
    pub fn new(input_size: usize) -> Self {
        LinearModel {
            weights: vec![0.0; input_size],
            bias: 0.0,
            input_size,
        }
    }
}

impl LinearModel {
    pub fn save_json(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_json(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let model: LinearModel = serde_json::from_str(&content)?;
        Ok(model)
    }

    pub fn save_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoded = bincode::serialize(self)?;
        std::fs::write(path, encoded)?;
        Ok(())
    }

    pub fn load_binary(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let model: LinearModel = bincode::deserialize(&data)?;
        Ok(model)
    }
}

impl Model for LinearModel {
    fn predict(&self, input: &Vector) -> Vector {
        let mut result = self.bias;
        for (i, item) in input.iter().enumerate().take(self.input_size) {
            result += self.weights[i] * *item;
        }
        vec![result]
    }

    fn train(&mut self, inputs: &[Vector], targets: &[Vector], config: &TrainConfig) {
        for epoch in 0..config.epochs {
            let mut total_loss = 0.0;
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let prediction = self.predict(input)[0];
                let target_val = target[0];
                let error = prediction - target_val;
                total_loss += error * error;

                for (i, item) in input.iter().enumerate().take(self.input_size) {
                    self.weights[i] -= config.learning_rate * 2.0 * error * *item;
                }
                self.bias -= config.learning_rate * 2.0 * error;
            }
            if epoch % 10 == 0 {
                println!(
                    "Epoque {} - Loss: {:.6}",
                    epoch,
                    total_loss / inputs.len() as f64
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predict() {
        let mut model = LinearModel::new(2);
        model.weights = vec![1.0, 2.0];
        model.bias = 1.0;
        let input = vec![3.0, 4.0];
        let result = model.predict(&input);
        assert_eq!(result[0], 12.0);
    }

    #[test]
    fn test_train_loss_diminue() {
        let mut model = LinearModel::new(2);
        let inputs = vec![vec![1.0, 2.0]];
        let targets = vec![vec![5.0]];
        let config = TrainConfig {
            learning_rate: 0.01,
            epochs: 100,
        };
        model.train(&inputs, &targets, &config);
        let result = model.predict(&inputs[0])[0];
        let ecart = (result - 5.0).abs();
        assert!(
            ecart < 0.1,
            "La prediction devrait etre proche de 5.0, got {}",
            result
        );
    }
}
