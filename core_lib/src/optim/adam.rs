use std::collections::HashMap;
use super::optimizer::Optimizer;

// Adam (Adaptive Moment Estimation)
// m_t = b1 * m_{t-1} + (1-b1) * grad
// v_t = b2 * v_{t-1} + (1-b2) * grad^2
// m_hat = m_t / (1 - b1^t)
// v_hat = v_t / (1 - b2^t)
// theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
pub struct Adam {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    m: HashMap<usize, f64>,
    v: HashMap<usize, f64>,
    t: HashMap<usize, usize>,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: HashMap::new(),
        }
    }

    pub fn with_betas(mut self, beta1: f64, beta2: f64) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    fn update(&mut self, idx: usize, value: f64, grad: f64) -> f64 {
        let t = {
            let t_entry = self.t.entry(idx).or_insert(0);
            *t_entry += 1;
            *t_entry as f64
        };

        let m_hat = {
            let m = self.m.entry(idx).or_insert(0.0);
            *m = self.beta1 * *m + (1.0 - self.beta1) * grad;
            *m / (1.0 - self.beta1.powf(t))
        };

        let v_hat = {
            let v = self.v.entry(idx).or_insert(0.0);
            *v = self.beta2 * *v + (1.0 - self.beta2) * grad * grad;
            *v / (1.0 - self.beta2.powf(t))
        };

        value - self.lr * m_hat / (v_hat.sqrt() + self.epsilon)
    }

    fn reset(&mut self) {
        self.m.clear();
        self.v.clear();
        self.t.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_reduces_param_for_positive_grad() {
        let mut adam = Adam::new(0.001);
        assert!(adam.update(0, 1.0, 1.0) < 1.0);
    }

    #[test]
    fn adam_increases_param_for_negative_grad() {
        let mut adam = Adam::new(0.001);
        assert!(adam.update(0, 0.0, -1.0) > 0.0);
    }

    #[test]
    fn adam_reset_clears_state() {
        let mut adam = Adam::new(0.001);
        for _ in 0..5 { adam.update(0, 1.0, 0.5); }
        adam.reset();
        let v_after_reset = adam.update(0, 1.0, 0.5);
        let mut adam2 = Adam::new(0.001);
        let v_fresh = adam2.update(0, 1.0, 0.5);
        assert!((v_after_reset - v_fresh).abs() < 1e-10);
    }

    #[test]
    fn adam_independent_params() {
        let mut adam = Adam::new(0.001);
        let v0 = adam.update(0, 1.0, 1.0);
        let v1 = adam.update(1, 1.0, -1.0);
        assert!(v0 < 1.0);
        assert!(v1 > 1.0);
    }

    #[test]
    fn adam_bias_correction_at_step_one() {
        let lr = 0.001;
        let mut adam = Adam::new(lr);
        let v1 = adam.update(0, 1.0, 1.0);
        assert!((1.0 - v1 - lr).abs() < 1e-6);
    }
}
