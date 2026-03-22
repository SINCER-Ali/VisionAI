use pyo3::prelude::*;
use core_lib::models::mlp::MLP;
use core_lib::math::vector::Vector;
use core_lib::optim::gradient_descent::GradientDescentConfig;

#[pyclass]
struct PyMLP {
    model: MLP,
}

#[pymethods]
impl PyMLP {
    #[new]
    fn new(layer_sizes: Vec<usize>) -> Self {
        PyMLP {
            model: MLP::new(&layer_sizes),
        }
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let v = Vector::from_vec(input);
        let result = self.model.predict(&v);
        result.data
    }

    fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        learning_rate: f64,
        epochs: usize,
    ) {
        let inputs: Vec<Vector> = inputs.into_iter().map(Vector::from_vec).collect();
        let targets: Vec<Vector> = targets.into_iter().map(Vector::from_vec).collect();
        let cfg = GradientDescentConfig { lr: learning_rate, epochs };
        self.model.train(&inputs, &targets, cfg);
    }
}

#[pymodule]
fn vision_ai(_m: &Bound<'_, PyModule>) -> PyResult<()> {
    _m.add_class::<PyMLP>()?;
    Ok(())
}