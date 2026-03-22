use crate::math::matrix::Matrix;
use crate::math::vector::Vector;
use crate::optim::gradient_descent::GradientDescentConfig;

#[derive(Clone, Debug)]
pub struct LinearRegression {
    pub w: Vector,
    pub b: f64,
}

impl LinearRegression {
    pub fn new(n_features: usize) -> Self {
        Self {
            w: Vector::new(n_features),
            b: 0.0,
        }
    }

    pub fn predict(&self, x: &Matrix) -> Vector {
        assert_eq!(x.cols, self.w.len, "X.cols must match w.len");
        let mut out = Vector::new(x.rows);

        for i in 0..x.rows {
            let mut s = self.b;
            for j in 0..x.cols {
                s += x.data[i * x.cols + j] * self.w.data[j];
            }
            out.data[i] = s;
        }
        out
    }

    pub fn mse_loss(y_hat: &Vector, y: &Vector) -> f64 {
        assert_eq!(y_hat.len, y.len, "y_hat and y must have same length");
        let mut sum = 0.0;
        for i in 0..y.len {
            let e = y_hat.data[i] - y.data[i];
            sum += e * e;
        }
        sum / (y.len as f64)
    }

    pub fn fit(&mut self, x: &Matrix, y: &Vector, cfg: GradientDescentConfig) {
        assert_eq!(x.rows, y.len, "X.rows must match y.len");
        assert_eq!(x.cols, self.w.len, "X.cols must match w.len");
        let n = x.rows as f64;
        let d = x.cols;

        for _epoch in 0..cfg.epochs {
            let y_hat = self.predict(x);

            let mut e = Vector::new(y.len);
            for i in 0..y.len {
                e.data[i] = y_hat.data[i] - y.data[i];
            }

            let scale = 1.0 / n;

            let mut grad_w = vec![0.0; d];
            for (j, gw) in grad_w.iter_mut().enumerate() {
                let mut s = 0.0;
                for i in 0..x.rows {
                    s += x.data[i * x.cols + j] * e.data[i];
                }
                *gw = scale * s;
            }

            let mut grad_b = 0.0;
            for i in 0..e.len {
                grad_b += e.data[i];
            }
            grad_b *= scale;

            for (j, gw) in grad_w.iter().enumerate() {
                self.w.data[j] -= cfg.lr * gw;
            }
            self.b -= cfg.lr * grad_b;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_simple_1d_line() {
        let n = 200usize;
        let mut x_data = Vec::with_capacity(n);
        let mut y_data = Vec::with_capacity(n);

        for i in 0..n {
            let x = (i as f64) / 10.0;
            x_data.push(x);
            y_data.push(3.0 * x + 2.0);
        }

        let x = Matrix::from_flat(n, 1, x_data);
        let y = Vector::from_vec(y_data);

        let mut model = LinearRegression::new(1);
        model.fit(
            &x,
            &y,
            GradientDescentConfig {
                lr: 0.01,
                epochs: 2000,
            },
        );

        assert!(
            (model.w.data[0] - 3.0).abs() < 1e-1,
            "w was {}",
            model.w.data[0]
        );
        assert!((model.b - 2.0).abs() < 1e-1, "b was {}", model.b);
    }
}
