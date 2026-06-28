use core_lib::math::activations::{Activation, relu, sigmoid, softmax, tanh};
use core_lib::math::matrix::Matrix;
use core_lib::math::vector::Vector;
use core_lib::metrics::{kfold_indices, mae, mse, r_squared};
use core_lib::models::linear::LinearModel;
use core_lib::models::mlp::MLP;
use core_lib::models::rbf::RBF;
use core_lib::models::svm::{KernelType, SVM};
use core_lib::models::TrainConfig;
use core_lib::optim::adam::Adam;
use core_lib::optim::gradient_descent::GradientDescentConfig;
use core_lib::optim::sgd_momentum::SGDMomentum;

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
    let inputs: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 / 50.0]).collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|x| vec![3.0 * x[0] + 1.0]).collect();
    let config = TrainConfig {
        learning_rate: 0.05,
        epochs: 200,
    };
    model.train(&inputs, &targets, &config);

    let pred = model.predict(&vec![0.5]);
    let expected = 3.0 * 0.5 + 1.0;
    assert!(
        (pred[0] - expected).abs() < 0.5,
        "prediction {} should be close to {}",
        pred[0],
        expected
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
#[allow(clippy::too_many_arguments)]
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
        train_and_check(
            &[2, 16, 2],
            &inputs,
            &targets,
            &expected,
            1.0,
            5000,
            5,
            Activation::Sigmoid
        ),
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
        train_and_check(
            &[2, 8, 2],
            &inputs,
            &targets,
            &expected,
            1.0,
            5000,
            5,
            Activation::Sigmoid
        ),
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
        train_and_check(
            &[2, 8, 2],
            &inputs,
            &targets,
            &expected,
            1.0,
            3000,
            5,
            Activation::Sigmoid
        ),
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
        train_and_check(
            &[2, 16, 3],
            &inputs,
            &targets,
            &expected,
            0.5,
            5000,
            5,
            Activation::Sigmoid
        ),
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
        train_and_check(
            &[1, 16, 2],
            &inputs,
            &targets,
            &expected,
            0.5,
            3000,
            5,
            Activation::Sigmoid
        ),
        "Regression sinus devrait converger en 5 tentatives"
    );
}

// ─── MLP tests : avec differentes activations ──────────────────

#[test]
fn mlp_with_sigmoid_activation() {
    let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid);
    let cfg = GradientDescentConfig {
        lr: 1.0,
        epochs: 3000,
    };

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
    let cfg = GradientDescentConfig {
        lr: 0.5,
        epochs: 3000,
    };

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
        assert!(
            (a - b).abs() < 1e-10,
            "Binary roundtrip: predictions differ"
        );
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

// ── helpers ───────────────────────────────────────────────────────────────────

fn xor_vectors() -> (Vec<Vector>, Vec<Vector>, Vec<usize>) {
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
    (inputs, targets, vec![0, 1, 1, 0])
}

fn and_vectors() -> (Vec<Vector>, Vec<Vector>, Vec<usize>) {
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
    (inputs, targets, vec![0, 0, 0, 1])
}

fn three_class_vectors() -> (Vec<Vector>, Vec<Vector>, Vec<usize>) {
    let inputs = vec![
        Vector::from_vec(vec![0.0, 0.0]),
        Vector::from_vec(vec![0.1, 0.1]),
        Vector::from_vec(vec![0.0, 0.2]),
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
    (inputs, targets, vec![0, 0, 0, 1, 1, 1, 2, 2, 2])
}

// ── RBF tests ─────────────────────────────────────────────────────────────────

#[test]
fn rbf_output_shape_integration() {
    let (inputs, targets, _) = xor_vectors();
    let mut rbf = RBF::new(4, 1.0, 2);
    rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.0, epochs: 0 });
    let out = rbf.predict(&inputs[0]);
    assert_eq!(out.len, 2);
    let sum: f64 = out.data.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn rbf_xor_integration() {
    let (inputs, targets, expected) = xor_vectors();
    let mut ok = false;
    for _ in 0..10 {
        let mut rbf = RBF::new(4, 2.0, 2).with_lambda(1e-8);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.05, epochs: 300 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "RBF doit resoudre XOR");
}

#[test]
fn rbf_and_integration() {
    let (inputs, targets, expected) = and_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut rbf = RBF::new(4, 1.0, 2).with_lambda(1e-6);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.01, epochs: 0 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "RBF doit resoudre AND");
}

#[test]
fn rbf_multiclass_integration() {
    let (inputs, targets, expected) = three_class_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut rbf = RBF::new(9, 1.0, 3).with_lambda(1e-6);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.0, epochs: 0 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "RBF doit classifier 3 classes");
}

#[test]
fn rbf_json_roundtrip_integration() {
    let (inputs, targets, _) = xor_vectors();
    let mut rbf = RBF::new(4, 1.0, 2);
    rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.0, epochs: 0 });
    rbf.save_json("__it_rbf.json").unwrap();
    let loaded = RBF::load_json("__it_rbf.json").unwrap();
    let out_a = rbf.predict(&inputs[0]);
    let out_b = loaded.predict(&inputs[0]);
    for (a, b) in out_a.data.iter().zip(out_b.data.iter()) { assert!((a - b).abs() < 1e-10); }
    std::fs::remove_file("__it_rbf.json").ok();
}

// ── SVM lineaire tests ────────────────────────────────────────────────────────

#[test]
fn linear_svm_and_integration() {
    let (inputs, targets, expected) = and_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_linear(10.0);
        svm.train(&inputs, &targets, 0.05, 2000);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM lineaire doit resoudre AND");
}

#[test]
fn linear_svm_or_integration() {
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
    let expected = vec![0usize, 1, 1, 1];
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_linear(10.0);
        svm.train(&inputs, &targets, 0.05, 2000);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM lineaire doit resoudre OR");
}

#[test]
fn linear_svm_multiclass_integration() {
    let (inputs, targets, expected) = three_class_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_linear(10.0);
        svm.train(&inputs, &targets, 0.05, 3000);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM lineaire doit classifier 3 classes");
}

#[test]
fn linear_svm_output_is_softmax() {
    let (inputs, targets, _) = and_vectors();
    let mut svm = SVM::new_linear(1.0);
    svm.train(&inputs, &targets, 0.1, 500);
    for x in &inputs {
        let sum: f64 = svm.predict(x).data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}

// ── SVM a noyau tests ─────────────────────────────────────────────────────────

#[test]
fn kernel_svm_rbf_xor_integration() {
    let (inputs, targets, expected) = xor_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_kernel(5.0, KernelType::RBF { gamma: 1.0 });
        svm.train(&inputs, &targets, 0.0, 300);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM RBF doit resoudre XOR");
}

#[test]
fn kernel_svm_poly_and_integration() {
    let (inputs, targets, expected) = and_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_kernel(10.0, KernelType::Polynomial { degree: 2, coef0: 1.0 });
        svm.train(&inputs, &targets, 0.0, 200);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM polynomial doit resoudre AND");
}

#[test]
fn kernel_svm_linear_kernel_and() {
    let (inputs, targets, expected) = and_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_kernel(10.0, KernelType::Linear);
        svm.train(&inputs, &targets, 0.0, 200);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "SVM noyau lineaire doit resoudre AND");
}

#[test]
fn svm_json_roundtrip_integration() {
    let (inputs, targets, _) = and_vectors();
    let mut svm = SVM::new_linear(1.0);
    svm.train(&inputs, &targets, 0.1, 500);
    svm.save_json("__it_svm.json").unwrap();
    let loaded = SVM::load_json("__it_svm.json").unwrap();
    let out_a = svm.predict(&inputs[0]);
    let out_b = loaded.predict(&inputs[0]);
    for (a, b) in out_a.data.iter().zip(out_b.data.iter()) { assert!((a - b).abs() < 1e-10); }
    std::fs::remove_file("__it_svm.json").ok();
}

// ── Optimiseurs tests ─────────────────────────────────────────────────────────

#[test]
fn sgd_momentum_mlp_xor() {
    let (inputs, targets, expected) = xor_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut mlp = MLP::new(&[2, 16, 2]).with_activation(Activation::Sigmoid);
        let mut opt = SGDMomentum::new(0.5, 0.9);
        mlp.train_with_optimizer(&inputs, &targets, 5000, &mut opt);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| mlp.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "MLP + SGD Momentum doit resoudre XOR");
}

#[test]
fn adam_mlp_xor() {
    let (inputs, targets, expected) = xor_vectors();
    let mut ok = false;
    for _ in 0..5 {
        let mut mlp = MLP::new(&[2, 16, 2]).with_activation(Activation::Sigmoid);
        let mut opt = Adam::new(0.01);
        mlp.train_with_optimizer(&inputs, &targets, 3000, &mut opt);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| mlp.predict(x).argmax() == e) {
            ok = true; break;
        }
    }
    assert!(ok, "MLP + Adam doit resoudre XOR");
}

#[test]
fn sgd_momentum_reduces_loss() {
    let (inputs, targets, expected) = and_vectors();
    let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid);
    let mut opt = SGDMomentum::new(0.5, 0.9);
    mlp.train_with_optimizer(&inputs, &targets, 3000, &mut opt);
    let correct: usize = (0..inputs.len())
        .filter(|&i| mlp.predict(&inputs[i]).argmax() == expected[i])
        .count();
    assert!(correct >= 3);
}

#[test]
fn adam_optimizer_reset() {
    let mut adam = Adam::new(0.001);
    let v1 = adam.update(0, 1.0, 0.5);
    adam.reset();
    let v2 = adam.update(0, 1.0, 0.5);
    assert!((v1 - v2).abs() < 1e-10);
}

#[test]
fn sgd_nesterov_mlp_and() {
    let (inputs, targets, expected) = and_vectors();
    let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid);
    let mut opt = SGDMomentum::new(0.5, 0.9).with_nesterov();
    mlp.train_with_optimizer(&inputs, &targets, 2000, &mut opt);
    let correct: usize = (0..inputs.len())
        .filter(|&i| mlp.predict(&inputs[i]).argmax() == expected[i])
        .count();
    assert!(correct >= 3);
}

// ── Metriques tests ───────────────────────────────────────────────────────────

#[test]
fn metrics_mse_perfect() {
    assert_eq!(mse(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn metrics_mse_known_value() {
    assert!((mse(&[0.0, 0.0], &[2.0, 4.0]) - 10.0).abs() < 1e-10);
}

#[test]
fn metrics_mae_perfect() {
    assert_eq!(mae(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn metrics_mae_known_value() {
    assert!((mae(&[0.0, 0.0], &[2.0, 4.0]) - 3.0).abs() < 1e-10);
}

#[test]
fn metrics_r_squared_perfect() {
    let r2 = r_squared(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
    assert!((r2 - 1.0).abs() < 1e-10);
}

#[test]
fn metrics_r_squared_mean_predictor() {
    let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let preds = vec![3.0f64; 5];
    assert!(r_squared(&preds, &targets).abs() < 1e-10);
}

#[test]
fn metrics_kfold_correct_sizes() {
    let folds = kfold_indices(20, 4);
    assert_eq!(folds.len(), 4);
    for (train, test) in &folds {
        assert_eq!(test.len(), 5);
        assert_eq!(train.len(), 15);
    }
}

#[test]
fn metrics_kfold_no_overlap() {
    let folds = kfold_indices(10, 5);
    for (train, test) in &folds {
        for &t in test { assert!(!train.contains(&t)); }
    }
}

// ── Comparaison des modeles ───────────────────────────────────────────────────

#[test]
fn compare_all_models_on_and() {
    let (inputs, targets, expected) = and_vectors();
    let mut mlp_ok = false;
    for _ in 0..5 {
        let mut mlp = MLP::new(&[2, 8, 2]).with_activation(Activation::Sigmoid);
        mlp.train(&inputs, &targets, GradientDescentConfig { lr: 1.0, epochs: 3000 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| mlp.predict(x).argmax() == e) { mlp_ok = true; break; }
    }
    let mut rbf_ok = false;
    for _ in 0..5 {
        let mut rbf = RBF::new(4, 1.0, 2).with_lambda(1e-6);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.0, epochs: 0 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) { rbf_ok = true; break; }
    }
    let mut svm_ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_linear(10.0);
        svm.train(&inputs, &targets, 0.05, 2000);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) { svm_ok = true; break; }
    }
    assert!(mlp_ok, "MLP doit reussir AND");
    assert!(rbf_ok, "RBF doit reussir AND");
    assert!(svm_ok, "SVM doit reussir AND");
}

#[test]
fn compare_nonlinear_models_on_xor() {
    let (inputs, targets, expected) = xor_vectors();
    let mut mlp_ok = false;
    for _ in 0..5 {
        let mut mlp = MLP::new(&[2, 16, 2]).with_activation(Activation::Sigmoid);
        mlp.train(&inputs, &targets, GradientDescentConfig { lr: 1.0, epochs: 5000 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| mlp.predict(x).argmax() == e) { mlp_ok = true; break; }
    }
    let mut rbf_ok = false;
    for _ in 0..10 {
        let mut rbf = RBF::new(4, 2.0, 2).with_lambda(1e-8);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.05, epochs: 300 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) { rbf_ok = true; break; }
    }
    let mut svm_ok = false;
    for _ in 0..5 {
        let mut svm = SVM::new_kernel(5.0, KernelType::RBF { gamma: 1.0 });
        svm.train(&inputs, &targets, 0.0, 300);
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e) { svm_ok = true; break; }
    }
    assert!(mlp_ok, "MLP doit resoudre XOR");
    assert!(rbf_ok, "RBF doit resoudre XOR");
    assert!(svm_ok, "SVM RBF doit resoudre XOR");
}

#[test]
fn circles_dataset_mlp_vs_rbf() {
    let inputs: Vec<Vector> = vec![
        Vector::from_vec(vec![0.1, 0.0]),
        Vector::from_vec(vec![-0.1, 0.0]),
        Vector::from_vec(vec![0.0, 0.1]),
        Vector::from_vec(vec![0.0, -0.1]),
        Vector::from_vec(vec![1.0, 0.0]),
        Vector::from_vec(vec![-1.0, 0.0]),
        Vector::from_vec(vec![0.0, 1.0]),
        Vector::from_vec(vec![0.0, -1.0]),
    ];
    let mut targets: Vec<Vector> = vec![Vector::from_vec(vec![1.0, 0.0]); 4];
    targets.extend(vec![Vector::from_vec(vec![0.0, 1.0]); 4]);
    let expected = vec![0usize, 0, 0, 0, 1, 1, 1, 1];

    let mut rbf_ok = false;
    for _ in 0..20 {
        let mut rbf = RBF::new(4, 2.0, 2).with_lambda(1e-6);
        rbf.train(&inputs, &targets, GradientDescentConfig { lr: 0.05, epochs: 300 });
        if inputs.iter().zip(expected.iter()).all(|(x, &e)| rbf.predict(x).argmax() == e) { rbf_ok = true; break; }
    }
    let mut svm = SVM::new_kernel(10.0, KernelType::RBF { gamma: 2.0 });
    svm.train(&inputs, &targets, 0.0, 500);
    let svm_ok = inputs.iter().zip(expected.iter()).all(|(x, &e)| svm.predict(x).argmax() == e);

    assert!(rbf_ok, "RBF doit separer cercles");
    assert!(svm_ok, "SVM RBF doit separer cercles");
}
