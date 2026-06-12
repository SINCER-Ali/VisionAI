use std::collections::HashMap;
use super::optimizer::Optimizer;

// SGD avec momentum
// v_t = momentum * v_{t-1} - lr * grad
// theta = theta + v_t
// Variante Nesterov : theta = theta + momentum * v_t - lr * grad
pub struct SGDMomentum {
    pub lr: f64,
    pub momentum: f64,
    pub nesterov: bool,
    velocities: HashMap<usize, f64>,
}

impl SGDMomentum {
    pub fn new(lr: f64, momentum: f64) -> Self {
        SGDMomentum { lr, momentum, nesterov: false, velocities: HashMap::new() }
    }

    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }
}

impl Optimizer for SGDMomentum {
    fn update(&mut self, idx: usize, value: f64, grad: f64) -> f64 {
        let v = self.velocities.entry(idx).or_insert(0.0);
        let v_prev = *v;
        *v = self.momentum * v_prev - self.lr * grad;

        if self.nesterov {
            value + self.momentum * *v - self.lr * grad
        } else {
            value + *v
        }
    }

    fn reset(&mut self) {
        self.velocities.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_momentum_reduces_param() {
        let mut opt = SGDMomentum::new(0.1, 0.9);
        let v1 = opt.update(0, 1.0, 1.0);
        assert!(v1 < 1.0);
    }

    #[test]
    fn sgd_momentum_accumulates_velocity() {
        let mut opt = SGDMomentum::new(0.1, 0.9);
        let mut v = 1.0f64;
        for _ in 0..5 { v = opt.update(0, v, 0.1); }
        let drop_5 = 1.0 - v;
        let mut opt2 = SGDMomentum::new(0.1, 0.9);
        let drop_1 = 1.0 - opt2.update(0, 1.0, 0.1);
        assert!(drop_5 > drop_1);
    }

    #[test]
    fn sgd_momentum_reset_clears_velocity() {
        let mut opt = SGDMomentum::new(0.1, 0.9);
        opt.update(0, 1.0, 1.0);
        opt.reset();
        let v_after_reset = opt.update(0, 1.0, 1.0);
        let mut opt2 = SGDMomentum::new(0.1, 0.9);
        let v_fresh = opt2.update(0, 1.0, 1.0);
        assert!((v_after_reset - v_fresh).abs() < 1e-10);
    }

    #[test]
    fn sgd_nesterov_differs_from_standard() {
        let mut standard = SGDMomentum::new(0.01, 0.9);
        let mut nesterov = SGDMomentum::new(0.01, 0.9).with_nesterov();
        let v_std = standard.update(0, 1.0, 1.0);
        let v_nes = nesterov.update(0, 1.0, 1.0);
        assert!((v_std - v_nes).abs() > 1e-12);
    }
}
