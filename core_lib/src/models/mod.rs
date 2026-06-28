pub type Vector = Vec<f64>;

pub struct TrainConfig {
    pub learning_rate: f64,
    pub epochs: usize,
}

pub trait Model {
    fn predict(&self, input: &Vector) -> Vector;

    fn train(&mut self, inputs: &[Vector], targets: &[Vector], config: &TrainConfig);
}

pub mod linear;
pub mod mlp;
pub mod rbf;
pub mod svm;
