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
