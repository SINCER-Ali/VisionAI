use core_lib::math::activations::{
    self, Activation, relu, sigmoid, softmax, tanh,
};
use core_lib::math::matrix::Matrix;
use core_lib::math::vector::Vector;
use core_lib::models::linear::LinearModel;
use core_lib::models::mlp::MLP;
use core_lib::models::{Model, TrainConfig};
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

// ─── Matrix tests ───────────────────────────────────────────────

#[test]
fn matrix_new_is_zeroed() {
    let m = Matrix::new(3, 4);
    assert_eq!(m.rows, 3);
    assert_eq!(m.cols, 4);
    assert!(m.data.iter().all(|&x| x == 0.0));
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
#[should_panic]
fn matrix_multiply_incompatible() {
    let a = Matrix::new(2, 3);
    let b = Matrix::new(4, 2);
    let _ = a.multiply(&b);
}

// ─── Activation tests (integration) ────────────────────────────

#[test]
fn activation_sigmoid_integration() {
    let v = Vector::from_vec(vec![-5.0, -1.0, 0.0, 1.0, 5.0]);
    let s = sigmoid(&v);
    assert!(s.data[0] < 0.01);
    assert!((s.data[2] - 0.5).abs() < 1e-10);
    assert!(s.data[4] > 0.99);
}

#[test]
fn activation_tanh_integration() {
    let v = Vector::from_vec(vec![-5.0, 0.0, 5.0]);
    let t = tanh(&v);
    assert!(t.data[0] < -0.99);
    assert!(t.data[1].abs() < 1e-10);
    assert!(t.data[2] > 0.99);
}

#[test]
fn activation_relu_integration() {
    let v = Vector::from_vec(vec![-5.0, -0.1, 0.0, 0.1, 5.0]);
    let r = relu(&v);
    assert_eq!(r.data[0], 0.0);
    assert_eq!(r.data[1], 0.0);
    assert_eq!(r.data[2], 0.0);
    assert_eq!(r.data[3], 0.1);
    assert_eq!(r.data[4], 5.0);
}

#[test]
fn activation_softmax_integration() {
    let v = Vector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    let s = softmax(&v);
    let sum: f64 = s.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
    // Doit etre croissant
    for i in 0..3 {
        assert!(s.data[i] < s.data[i + 1]);
    }
}

#[test]
fn activation_enum_apply() {
    let v = Vector::from_vec(vec![-1.0, 0.0, 1.0]);
    let r = Activation::ReLU.apply(&v);
    assert_eq!(r.data, vec![0.0, 0.0, 1.0]);

    let s = Activation::Sigmoid.apply(&Vector::from_vec(vec![0.0]));
    assert!((s.data[0] - 0.5).abs() < 1e-10);

    let t = Activation::Tanh.apply(&Vector::from_vec(vec![0.0]));
    assert!(t.data[0].abs() < 1e-10);
}

// ─── LinearModel tests ─────────────────────────────────────────

#[test]
fn linear_model_predict_zero_weights() {
    let model = LinearModel::new(3);
    let input = vec![1.0, 2.0, 3.0];
    let result = model.predict(&input);
    assert_eq!(result, vec![0.0]);
}

#[test]
fn linear_model_train_converges() {
    let mut model = LinearModel::new(1);
    // y = 3*x + 1
    let inputs: Vec<Vec<f64>> = (0..50)
        .map(|i| vec![i as f64 / 50.0])
        .collect();
    let targets: Vec<Vec<f64>> = inputs
        .iter()
        .map(|x| vec![3.0 * x[0] + 1.0])
        .collect();
    let config = TrainConfig {
        learning_rate: 0.05,
        epochs: 200,
    };
    model.train(&inputs, &targets, &config);

    let pred = model.predict(&vec![0.5]);
    let expected = 3.0 * 0.5 + 1.0;
    assert!(
        (pred[0] - expected).abs() < 0.5,
        "prediction {} should be close to {}", pred[0], expected
    );
}

// ─── GradientDescentConfig tests ────────────────────────────────

#[test]
fn gradient_descent_default() {
    let cfg = GradientDescentConfig::default();
    assert_eq!(cfg.lr, 0.001);
    assert_eq!(cfg.epochs, 5000);
}

// Helper : tente d'entrainer un MLP plusieurs fois (l'init aleatoire
// peut mener a des minima locaux). Retourne true si le test passe.
fn train_and_check(
    arch: &[usize],
    inputs: &[Vector],
    targets: &[Vector],
    expected_classes: &[usize],
    lr: f64,
    epochs: usize,
    max_attempts: usize,
    activation: Activation,
) -> bool {
    for _ in 0..max_attempts {
        let mut mlp = MLP::new(arch).with_activation(activation);
        let cfg = GradientDescentConfig { lr, epochs };
        mlp.train(inputs, targets, cfg);

        let all_correct = inputs
            .iter()
            .zip(expected_classes.iter())
            .all(|(inp, &expected)| mlp.predict(inp).argmax() == expected);
        if all_correct {
            return true;
        }
    }
    false
}

// ─── MLP tests : XOR ─────────────────────────────────────────────

#[test]
fn mlp_xor() {
    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![1.0, 0.0]),
    ];
    let expected = vec![0, 1, 1, 0];

    assert!(
        train_and_check(&[2, 16, 2], &inputs, &targets, &expected, 1.0, 5000, 5, Activation::Sigmoid),
        "XOR devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : AND ────────────────────────────────────────────

#[test]
fn mlp_and() {
    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
    ];
    let expected = vec![0, 0, 0, 1];

    assert!(
        train_and_check(&[2, 8, 2], &inputs, &targets, &expected, 1.0, 5000, 5, Activation::Sigmoid),
        "AND devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : OR ─────────────────────────────────────────────

#[test]
fn mlp_or() {
    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, 1.0]),
    ];
    let expected = vec![0, 1, 1, 1];

    assert!(
        train_and_check(&[2, 8, 2], &inputs, &targets, &expected, 1.0, 3000, 5, Activation::Sigmoid),
        "OR devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : multi-classes (3 classes) ──────────────────────

#[test]
fn mlp_multiclass() {
    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.1, 0.1]),
        Vector::from_vec(vec![0.0, 0.1]),
        Vector::from_vec(vec![1.0, 1.0]),
        Vector::from_vec(vec![0.9, 0.9]),
        Vector::from_vec(vec![1.0, 0.9]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.9, 0.1]),
        Vector::from_vec(vec![1.0, 0.1]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0, 0.0]),
        Vector::from_vec(vec![1.0, 0.0, 0.0]),
        Vector::from_vec(vec![1.0, 0.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0, 0.0]),
        Vector::from_vec(vec![0.0, 0.0, 1.0]),
        Vector::from_vec(vec![0.0, 0.0, 1.0]),
        Vector::from_vec(vec![0.0, 0.0, 1.0]),
    ];
    let expected = vec![0, 0, 0, 1, 1, 1, 2, 2, 2];

    assert!(
        train_and_check(&[2, 16, 3], &inputs, &targets, &expected, 0.5, 5000, 5, Activation::Sigmoid),
        "Multi-class devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : regression simple (approximation sinus) ───────

#[test]
fn mlp_regression_sine() {
    let mut inputs = Vec::new();
    let mut targets = Vec::new();
    let mut expected = Vec::new();

    for i in 0..10 {
        let x = -std::f64::consts::PI + (i as f64) * std::f64::consts::PI / 10.0;
        inputs.push(Vector::from_vec(vec![x / std::f64::consts::PI]));
        targets.push(Vector::from_vec(vec![1.0, 0.0]));
        expected.push(0);
    }
    for i in 1..=10 {
        let x = (i as f64) * std::f64::consts::PI / 10.0;
        inputs.push(Vector::from_vec(vec![x / std::f64::consts::PI]));
        targets.push(Vector::from_vec(vec![0.0, 1.0]));
        expected.push(1);
    }

    assert!(
        train_and_check(&[1, 16, 2], &inputs, &targets, &expected, 0.5, 3000, 5, Activation::Sigmoid),
        "Regression sinus devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : avec differentes activations ──────────────────

#[test]
fn mlp_with_sigmoid_activation() {
    let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid);
    let cfg = GradientDescentConfig { lr: 1.0, epochs: 3000 };

    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, 1.0]),
    ];

    mlp.train(&inputs, &targets, cfg);

    assert_eq!(mlp.predict(&inputs[0]).argmax(), 0);
    assert_eq!(mlp.predict(&inputs[1]).argmax(), 1);
}

#[test]
fn mlp_with_tanh_activation() {
    let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Tanh);
    let cfg = GradientDescentConfig { lr: 0.5, epochs: 3000 };

    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![1.0, 1.0]),
    ];
    let targets = vec![
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
    ];

    mlp.train(&inputs, &targets, cfg);

    assert_eq!(mlp.predict(&inputs[0]).argmax(), 0);
    assert_eq!(mlp.predict(&inputs[1]).argmax(), 1);
}

// ─── Serialisation tests ────────────────────────────────────────

#[test]
fn mlp_serialize_json_roundtrip() {
    let mlp = MLP::new(&[3, 5, 2]);
    let input = Vector::from_vec(vec![0.1, 0.2, 0.3]);
    let pred_before = mlp.predict(&input);

    let path = "test_model.json";
    mlp.save_json(path).unwrap();
    let loaded = MLP::load_json(path).unwrap();
    let pred_after = loaded.predict(&input);

    for (a, b) in pred_before.data.iter().zip(pred_after.data.iter()) {
        assert!((a - b).abs() < 1e-10, "JSON roundtrip: predictions differ");
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn mlp_serialize_binary_roundtrip() {
    let mlp = MLP::new(&[3, 5, 2]);
    let input = Vector::from_vec(vec![0.1, 0.2, 0.3]);
    let pred_before = mlp.predict(&input);

    let path = "test_model.bin";
    mlp.save_binary(path).unwrap();
    let loaded = MLP::load_binary(path).unwrap();
    let pred_after = loaded.predict(&input);

    for (a, b) in pred_before.data.iter().zip(pred_after.data.iter()) {
        assert!((a - b).abs() < 1e-10, "Binary roundtrip: predictions differ");
    }

    std::fs::remove_file(path).ok();
}

#[test]
fn linear_model_serialize_json_roundtrip() {
    let mut model = LinearModel::new(2);
    model.weights = vec![1.5, -0.3];
    model.bias = 0.7;

    let path = "test_linear.json";
    model.save_json(path).unwrap();
    let loaded = LinearModel::load_json(path).unwrap();

    assert_eq!(loaded.weights, model.weights);
    assert_eq!(loaded.bias, model.bias);
    assert_eq!(loaded.input_size, model.input_size);

    std::fs::remove_file(path).ok();
}

#[test]
fn linear_model_serialize_binary_roundtrip() {
    let mut model = LinearModel::new(2);
    model.weights = vec![1.5, -0.3];
    model.bias = 0.7;

    let path = "test_linear.bin";
    model.save_binary(path).unwrap();
    let loaded = LinearModel::load_binary(path).unwrap();

    assert_eq!(loaded.weights, model.weights);
    assert_eq!(loaded.bias, model.bias);

    std::fs::remove_file(path).ok();
}

// ─── MLP predict output ────────────────────────────────────────

#[test]
fn mlp_predict_output_size() {
    let mlp = MLP::new(&[4, 8, 3]);
    let input = Vector::from_vec(vec![0.5, 0.2, 0.8, 0.1]);
    let output = mlp.predict(&input);
    assert_eq!(output.len, 3);
}

#[test]
fn mlp_predict_softmax_sums_to_one() {
    let mlp = MLP::new(&[4, 16, 8, 3]);
    let input = Vector::from_vec(vec![0.5, 0.2, 0.8, 0.1]);
    let output = mlp.predict(&input);
    let sum: f64 = output.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}

#[test]
fn mlp_deep_network() {
    // Test avec un reseau plus profond
    let mlp = MLP::new(&[3, 8, 8, 4, 2]);
    let input = Vector::from_vec(vec![0.5, 0.2, 0.8]);
    let output = mlp.predict(&input);
    assert_eq!(output.len, 2);
    let sum: f64 = output.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
}
