use core_lib::math::matrix::Matrix;
use core_lib::math::vector::Vector;
use core_lib::models::linear::LinearRegression;
use core_lib::optim::gradient_descent::GradientDescentConfig;

// ─── Vector tests ───────────────────────────────────────────────

#[test]
fn vector_new_is_zeroed() {
    let v = Vector::new(5);
    assert_eq!(v.len, 5);
    assert!(v.data.iter().all(|&x| x == 0.0));
}

#[test]
fn vector_from_vec_roundtrip() {
    let data = vec![1.0, 2.0, 3.0];
    let v = Vector::from_vec(data.clone());
    assert_eq!(v.len, 3);
    assert_eq!(v.data, data);
}

#[test]
fn vector_get_set() {
    let mut v = Vector::new(3);
    v.set(1, 42.0);
    assert_eq!(v.get(1), 42.0);
    assert_eq!(v.get(0), 0.0);
}

#[test]
fn vector_random_has_correct_length() {
    let v = Vector::random(10);
    assert_eq!(v.len, 10);
    assert_eq!(v.data.len(), 10);
}

#[test]
fn vector_sum_and_mean() {
    let v = Vector::from_vec(vec![2.0, 4.0, 6.0]);
    assert_eq!(v.sum(), 12.0);
    assert_eq!(v.mean(), 4.0);
}

#[test]
fn vector_scale_by_zero() {
    let v = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let scaled = v.scale(0.0);
    assert!(scaled.data.iter().all(|&x| x == 0.0));
}

#[test]
fn vector_scale_negative() {
    let v = Vector::from_vec(vec![1.0, -2.0]);
    let s = v.scale(-1.0);
    assert_eq!(s.data, vec![-1.0, 2.0]);
}

#[test]
fn vector_dot_orthogonal() {
    let a = Vector::from_vec(vec![1.0, 0.0]);
    let b = Vector::from_vec(vec![0.0, 1.0]);
    assert_eq!(a.dot(&b), 0.0);
}

#[test]
fn vector_l2_norm_unit() {
    let v = Vector::from_vec(vec![1.0, 0.0, 0.0]);
    assert!((v.l2_norm() - 1.0).abs() < 1e-12);
}

#[test]
fn vector_argmax_first_element() {
    let v = Vector::from_vec(vec![99.0, 1.0, 2.0]);
    assert_eq!(v.argmax(), 0);
}

#[test]
fn vector_argmax_last_element() {
    let v = Vector::from_vec(vec![1.0, 2.0, 99.0]);
    assert_eq!(v.argmax(), 2);
}

#[test]
#[should_panic(expected = "Vector dimensions mismatch")]
fn vector_add_dimension_mismatch() {
    let a = Vector::new(2);
    let b = Vector::new(3);
    let _ = a.add(&b);
}

#[test]
#[should_panic(expected = "Vector dimensions mismatch")]
fn vector_sub_dimension_mismatch() {
    let a = Vector::new(2);
    let b = Vector::new(3);
    let _ = a.sub(&b);
}

#[test]
#[should_panic(expected = "Vector dimensions mismatch")]
fn vector_hadamard_dimension_mismatch() {
    let a = Vector::new(2);
    let b = Vector::new(3);
    let _ = a.hadamard(&b);
}

// ─── Matrix tests ───────────────────────────────────────────────

#[test]
fn matrix_new_is_zeroed() {
    let m = Matrix::new(3, 4);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
    assert!(m.data.iter().all(|&x| x == 0.0));
}

#[test]
fn matrix_from_flat_stores_correctly() {
    let m = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(m.data[0], 1.0);
    assert_eq!(m.data[5], 6.0);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn matrix_from_flat_wrong_size() {
    Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0]);
}

#[test]
fn matrix_random_has_correct_shape() {
    let m = Matrix::random(5, 3);
    assert_eq!(m.rows, 5);
    assert_eq!(m.cols, 3);
    assert_eq!(m.data.len(), 15);
}

#[test]
fn matrix_transpose_square() {
    let m = Matrix::from_flat(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let t = m.transpose();
    assert_eq!(t.data, vec![1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn matrix_transpose_rectangular() {
    let m = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let t = m.transpose();
    assert_eq!(t.rows, 3);
    assert_eq!(t.cols, 2);
    assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn matrix_double_transpose_is_identity() {
    let m = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let tt = m.transpose().transpose();
    assert_eq!(tt.rows, m.rows);
    assert_eq!(tt.cols, m.cols);
    assert_eq!(tt.data, m.data);
}

#[test]
fn matrix_multiply_identity() {
    let a = Matrix::from_flat(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let eye = Matrix::from_flat(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let result = a.multiply(&eye);
    assert_eq!(result.data, a.data);
}

#[test]
fn matrix_multiply_vector_column() {
    // Matrix (2x3) * column vector as Matrix (3x1)
    let a = Matrix::from_flat(2, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let b = Matrix::from_flat(3, 1, vec![5.0, 7.0, 9.0]);
    let r = a.multiply(&b);
    assert_eq!(r.rows, 2);
    assert_eq!(r.cols, 1);
    assert_eq!(r.data, vec![5.0, 7.0]);
}

#[test]
#[should_panic]
fn matrix_multiply_incompatible() {
    let a = Matrix::new(2, 3);
    let b = Matrix::new(4, 2);
    let _ = a.multiply(&b);
}

// ─── GradientDescentConfig tests ────────────────────────────────

#[test]
fn gradient_descent_default() {
    let cfg = GradientDescentConfig::default();
    assert_eq!(cfg.lr, 0.001);
    assert_eq!(cfg.epochs, 5000);
}

// ─── LinearRegression tests ────────────────────────────────────

#[test]
fn linear_regression_predict_zero_weights() {
    let model = LinearRegression::new(3);
    let x = Matrix::from_flat(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y_hat = model.predict(&x);
    assert_eq!(y_hat.len, 2);
    assert!(y_hat.data.iter().all(|&v| v == 0.0));
}

#[test]
fn linear_regression_mse_zero_error() {
    let y = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    assert_eq!(LinearRegression::mse_loss(&y, &y), 0.0);
}

#[test]
fn linear_regression_mse_known_value() {
    let y_hat = Vector::from_vec(vec![1.0, 2.0, 3.0]);
    let y = Vector::from_vec(vec![1.0, 2.0, 4.0]);
    // error = [0, 0, -1], mse = (0 + 0 + 1) / 3
    let mse = LinearRegression::mse_loss(&y_hat, &y);
    assert!((mse - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
#[should_panic]
fn linear_regression_predict_dimension_mismatch() {
    let model = LinearRegression::new(3);
    let x = Matrix::from_flat(2, 5, vec![0.0; 10]);
    let _ = model.predict(&x);
}

#[test]
fn linear_regression_fit_constant_function() {
    // y = 5.0 (constant), small x values to aid convergence
    let n = 100;
    let x_data: Vec<f64> = (0..n).map(|i| (i as f64) / 100.0).collect();
    let y_data: Vec<f64> = vec![5.0; n];

    let x = Matrix::from_flat(n, 1, x_data);
    let y = Vector::from_vec(y_data);

    let mut model = LinearRegression::new(1);
    model.fit(
        &x,
        &y,
        GradientDescentConfig {
            lr: 0.1,
            epochs: 5000,
        },
    );

    // w should be ~0, b should be ~5
    assert!(model.w.data[0].abs() < 1.0, "w={}", model.w.data[0]);
    assert!((model.b - 5.0).abs() < 1.0, "b={}", model.b);
}

#[test]
fn linear_regression_fit_2d() {
    // y = 2*x1 + 3*x2 + 1
    let n = 200;
    let mut x_data = Vec::with_capacity(n * 2);
    let mut y_data = Vec::with_capacity(n);

    for i in 0..n {
        let x1 = (i as f64) / 100.0;
        let x2 = (i as f64) / 200.0;
        x_data.push(x1);
        x_data.push(x2);
        y_data.push(2.0 * x1 + 3.0 * x2 + 1.0);
    }

    let x = Matrix::from_flat(n, 2, x_data);
    let y = Vector::from_vec(y_data);

    let mut model = LinearRegression::new(2);
    model.fit(
        &x,
        &y,
        GradientDescentConfig {
            lr: 0.01,
            epochs: 10000,
        },
    );

    // Check the model produces good predictions (low MSE) rather than exact weight recovery
    let y_hat = model.predict(&x);
    let mse = LinearRegression::mse_loss(&y_hat, &y);
    assert!(mse < 0.1, "mse={mse}");
}

#[test]
fn linear_regression_prediction_improves_after_training() {
    let n = 100;
    let x_data: Vec<f64> = (0..n).map(|i| (i as f64) / 10.0).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x = Matrix::from_flat(n, 1, x_data);
    let y = Vector::from_vec(y_data);

    let mut model = LinearRegression::new(1);

    let y_hat_before = model.predict(&x);
    let loss_before = LinearRegression::mse_loss(&y_hat_before, &y);

    model.fit(
        &x,
        &y,
        GradientDescentConfig {
            lr: 0.01,
            epochs: 500,
        },
    );

    let y_hat_after = model.predict(&x);
    let loss_after = LinearRegression::mse_loss(&y_hat_after, &y);

    assert!(
        loss_after < loss_before,
        "loss should decrease after training: before={loss_before}, after={loss_after}"
    );
}
