use super::vector::Vector;
use serde::{Deserialize, Serialize};

/// Sigmoid : 1 / (1 + e^(-x))
pub fn sigmoid(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
    Vector::from_vec(data)
}

/// Derivee de sigmoid : s(x) * (1 - s(x))
pub fn sigmoid_derivative(v: &Vector) -> Vector {
    let s = sigmoid(v);
    let data = s
        .data
        .iter()
        .map(|si| si * (1.0 - si))
        .collect();
    Vector::from_vec(data)
}

/// Tanh
pub fn tanh(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| x.tanh()).collect();
    Vector::from_vec(data)
}

/// Derivee de tanh : 1 - tanh(x)^2
pub fn tanh_derivative(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| {
        let t = x.tanh();
        1.0 - t * t
    }).collect();
    Vector::from_vec(data)
}

/// ReLU : max(0, x)
pub fn relu(v: &Vector) -> Vector {
    let data = v.data.iter().map(|x| x.max(0.0)).collect();
    Vector::from_vec(data)
}

/// Derivee de ReLU
pub fn relu_derivative(v: &Vector) -> Vector {
    let data = v
        .data
        .iter()
        .map(|x| if *x > 0.0 { 1.0 } else { 0.0 })
        .collect();
    Vector::from_vec(data)
}

/// Softmax : e^(xi - max) / sum(e^(xj - max))
pub fn softmax(v: &Vector) -> Vector {
    let max = v.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.data.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    Vector::from_vec(exps.iter().map(|x| x / sum).collect())
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    Sigmoid,
    Tanh,
    ReLU,
    Softmax,
}

impl Activation {
    pub fn apply(&self, v: &Vector) -> Vector {
        match self {
            Activation::Sigmoid => sigmoid(v),
            Activation::Tanh => tanh(v),
            Activation::ReLU => relu(v),
            Activation::Softmax => softmax(v),
        }
    }

    pub fn derivative(&self, v: &Vector) -> Vector {
        match self {
            Activation::Sigmoid => sigmoid_derivative(v),
            Activation::Tanh => tanh_derivative(v),
            Activation::ReLU => relu_derivative(v),
            Activation::Softmax => {
                // Pour la backprop avec cross-entropy, la derivee est geree
                // directement dans le calcul du delta (output - target)
                Vector::from_vec(vec![1.0; v.len])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid_bounds() {
        let v = Vector::from_vec(vec![-10.0, 0.0, 10.0]);
        let s = sigmoid(&v);
        for val in &s.data {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_sigmoid_at_zero() {
        let v = Vector::from_vec(vec![0.0]);
        let s = sigmoid(&v);
        assert!((s.data[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_sigmoid_derivative_at_zero() {
        let v = Vector::from_vec(vec![0.0]);
        let d = sigmoid_derivative(&v);
        assert!((d.data[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_tanh_bounds() {
        let v = Vector::from_vec(vec![-10.0, 0.0, 10.0]);
        let t = tanh(&v);
        for val in &t.data {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
    }

    #[test]
    fn test_tanh_at_zero() {
        let v = Vector::from_vec(vec![0.0]);
        let t = tanh(&v);
        assert!(t.data[0].abs() < 1e-10);
    }

    #[test]
    fn test_tanh_derivative_at_zero() {
        let v = Vector::from_vec(vec![0.0]);
        let d = tanh_derivative(&v);
        assert!((d.data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu() {
        let v = Vector::from_vec(vec![-2.0, 0.0, 3.0]);
        let r = relu(&v);
        assert_eq!(r.data, vec![0.0, 0.0, 3.0]);
    }

    #[test]
    fn test_relu_derivative() {
        let v = Vector::from_vec(vec![-2.0, 0.0, 3.0]);
        let d = relu_derivative(&v);
        assert_eq!(d.data, vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&v);
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_ordering() {
        let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
        let s = softmax(&v);
        assert!(s.data[0] < s.data[1]);
        assert!(s.data[1] < s.data[2]);
    }

    #[test]
    fn test_activation_enum() {
        let v = Vector::from_vec(vec![-1.0, 0.0, 1.0]);
        let r = Activation::ReLU.apply(&v);
        assert_eq!(r.data, vec![0.0, 0.0, 1.0]);

        let s = Activation::Sigmoid.apply(&Vector::from_vec(vec![0.0]));
        assert!((s.data[0] - 0.5).abs() < 1e-10);

        let t = Activation::Tanh.apply(&Vector::from_vec(vec![0.0]));
        assert!(t.data[0].abs() < 1e-10);
    }
}
