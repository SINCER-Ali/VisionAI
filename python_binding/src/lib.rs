use core_lib::math::matrix::Matrix as CoreMatrix;
use core_lib::math::vector::Vector as CoreVector;
use core_lib::models::linear::LinearModel;
use core_lib::models::mlp::MLP;
use core_lib::models::rbf::RBF;
use core_lib::models::svm::{KernelType, SVM};
use core_lib::models::{Model, TrainConfig};
use core_lib::optim::gradient_descent::GradientDescentConfig;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fs;

#[pyclass]
struct PyVector {
    inner: CoreVector,
}

#[pymethods]
impl PyVector {
    #[new]
    fn new(data: Vec<f64>) -> Self {
        Self {
            inner: CoreVector::from_vec(data),
        }
    }

    #[staticmethod]
    fn zeros(len: usize) -> Self {
        Self {
            inner: CoreVector::new(len),
        }
    }

    fn to_list(&self) -> Vec<f64> {
        self.inner.data.clone()
    }

    fn len(&self) -> usize {
        self.inner.len
    }

    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    fn mean(&self) -> PyResult<f64> {
        if self.inner.len == 0 {
            Err(pyo3::exceptions::PyValueError::new_err("empty vector"))
        } else {
            Ok(self.inner.mean())
        }
    }

    fn dot(&self, other: &PyVector) -> PyResult<f64> {
        if self.inner.len != other.inner.len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension mismatch",
            ));
        }
        Ok(self.inner.dot(&other.inner))
    }

    fn add(&self, other: &PyVector) -> PyResult<PyVector> {
        if self.inner.len != other.inner.len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension mismatch",
            ));
        }
        Ok(PyVector {
            inner: self.inner.add(&other.inner),
        })
    }

    fn sub(&self, other: &PyVector) -> PyResult<PyVector> {
        if self.inner.len != other.inner.len {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "dimension mismatch",
            ));
        }
        Ok(PyVector {
            inner: self.inner.sub(&other.inner),
        })
    }

    fn scale(&self, alpha: f64) -> PyVector {
        PyVector {
            inner: self.inner.scale(alpha),
        }
    }
}

#[pyclass]
struct PyMatrix {
    inner: CoreMatrix,
}

#[pymethods]
impl PyMatrix {
    #[new]
    #[pyo3(signature = (rows, cols, data=None))]
    fn new(rows: usize, cols: usize, data: Option<Vec<f64>>) -> PyResult<Self> {
        let inner = if let Some(data) = data {
            if data.len() != rows * cols {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "invalid matrix size",
                ));
            }
            CoreMatrix::from_flat(rows, cols, data)
        } else {
            CoreMatrix::new(rows, cols)
        };
        Ok(Self { inner })
    }

    fn rows(&self) -> usize {
        self.inner.rows
    }

    fn cols(&self) -> usize {
        self.inner.cols
    }

    fn to_list(&self) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(self.inner.rows);
        for r in 0..self.inner.rows {
            let start = r * self.inner.cols;
            let end = start + self.inner.cols;
            out.push(self.inner.data[start..end].to_vec());
        }
        out
    }

    fn transpose(&self) -> PyMatrix {
        PyMatrix {
            inner: self.inner.transpose(),
        }
    }
}

#[pyclass]
struct LinearRegression {
    model: LinearModel,
}

#[pymethods]
impl LinearRegression {
    #[new]
    fn new(input_size: usize) -> Self {
        Self {
            model: LinearModel::new(input_size),
        }
    }

    fn predict(&self, input: Vec<f64>) -> PyResult<Vec<f64>> {
        Ok(self.model.predict(&input))
    }

    fn train(
        &mut self,
        x: Vec<Vec<f64>>,
        y: Vec<Vec<f64>>,
        epochs: usize,
        lr: f64,
    ) -> PyResult<()> {
        let cfg = TrainConfig {
            learning_rate: lr,
            epochs,
        };
        self.model.train(&x, &y, &cfg);
        Ok(())
    }
}

#[pyclass]
struct PyRBF {
    model: RBF,
}

#[pymethods]
impl PyRBF {
    /// Signature historique conservée (input_size, output_size, n_centers, sigma)
    /// pour rester compatible avec les notebooks. En interne on délègue au RBF
    /// du projet : `sigma` est converti en `gamma` (gamma = 1 / (2*sigma^2)),
    /// et `input_size` est déduit automatiquement des données à l'entraînement.
    #[new]
    #[pyo3(signature = (input_size, output_size, n_centers=10, sigma=1.0))]
    fn new(input_size: usize, output_size: usize, n_centers: usize, sigma: f64) -> Self {
        let _ = input_size; // déduit des données dans train()
        let gamma = if sigma > 0.0 {
            1.0 / (2.0 * sigma * sigma)
        } else {
            1.0
        };
        Self {
            model: RBF::new(n_centers, gamma, output_size),
        }
    }

    /// Conservé pour compatibilité : ce RBF initialise ses centres
    /// automatiquement au début de `train()`, donc cet appel est sans effet.
    fn init_centers_random(&mut self, _data: Vec<Vec<f64>>) {}

    #[pyo3(signature = (inputs, targets, lr, epochs, regression=false))]
    fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        lr: f64,
        epochs: usize,
        regression: bool,
    ) {
        let _ = regression; // ce RBF fait de la classification (softmax)
        let inputs: Vec<CoreVector> = inputs.into_iter().map(CoreVector::from_vec).collect();
        let targets: Vec<CoreVector> = targets.into_iter().map(CoreVector::from_vec).collect();
        let cfg = GradientDescentConfig { lr, epochs };
        self.model.train(&inputs, &targets, cfg);
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let v = CoreVector::from_vec(input);
        self.model.predict(&v).data
    }

    fn accuracy(&self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
        let inputs: Vec<CoreVector> = inputs.into_iter().map(CoreVector::from_vec).collect();
        let targets: Vec<CoreVector> = targets.into_iter().map(CoreVector::from_vec).collect();
        if inputs.is_empty() {
            return 0.0;
        }
        let correct = inputs
            .iter()
            .zip(targets.iter())
            .filter(|(x, t)| self.model.predict(x).argmax() == t.argmax())
            .count();
        correct as f64 / inputs.len() as f64
    }

    fn save_json(&self, path: String) -> PyResult<()> {
        self.model
            .save_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load_json(path: String) -> PyResult<PyRBF> {
        let model = RBF::load_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyRBF { model })
    }
}

#[pyclass]
struct PySVM {
    model: SVM,
}

#[pymethods]
impl PySVM {
    /// kernel : "linear" | "rbf" | "poly"
    #[new]
    #[pyo3(signature = (c=1.0, kernel="linear", gamma=1.0, degree=3, coef0=1.0))]
    fn new(c: f64, kernel: &str, gamma: f64, degree: usize, coef0: f64) -> PyResult<Self> {
        let model = match kernel {
            "linear" => SVM::new_linear(c),
            "rbf" => SVM::new_kernel(c, KernelType::RBF { gamma }),
            "poly" | "polynomial" => SVM::new_kernel(c, KernelType::Polynomial { degree, coef0 }),
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "kernel inconnu '{other}' (attendu : linear | rbf | poly)"
                )));
            }
        };
        Ok(Self { model })
    }

    fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, lr: f64, epochs: usize) {
        let inputs: Vec<CoreVector> = inputs.into_iter().map(CoreVector::from_vec).collect();
        let targets: Vec<CoreVector> = targets.into_iter().map(CoreVector::from_vec).collect();
        self.model.train(&inputs, &targets, lr, epochs);
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let v = CoreVector::from_vec(input);
        self.model.predict(&v).data
    }

    fn accuracy(&self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
        let inputs: Vec<CoreVector> = inputs.into_iter().map(CoreVector::from_vec).collect();
        let targets: Vec<CoreVector> = targets.into_iter().map(CoreVector::from_vec).collect();
        if inputs.is_empty() {
            return 0.0;
        }
        let correct = inputs
            .iter()
            .zip(targets.iter())
            .filter(|(x, t)| self.model.predict(x).argmax() == t.argmax())
            .count();
        correct as f64 / inputs.len() as f64
    }

    fn save_json(&self, path: String) -> PyResult<()> {
        self.model
            .save_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load_json(path: String) -> PyResult<PySVM> {
        let model = SVM::load_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PySVM { model })
    }
}

#[pyclass]
struct PyMLP {
    model: MLP,
}

#[pymethods]
impl PyMLP {
    #[new]
    fn new(layer_sizes: Vec<usize>) -> Self {
        Self {
            model: MLP::new(&layer_sizes),
        }
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let v = CoreVector::from_vec(input);
        self.model.predict(&v).data
    }

    fn train(
        &mut self,
        inputs: Vec<Vec<f64>>,
        targets: Vec<Vec<f64>>,
        learning_rate: f64,
        epochs: usize,
    ) {
        let inputs: Vec<CoreVector> = inputs.into_iter().map(CoreVector::from_vec).collect();
        let targets: Vec<CoreVector> = targets.into_iter().map(CoreVector::from_vec).collect();
        let cfg = GradientDescentConfig {
            lr: learning_rate,
            epochs,
        };
        self.model.train(&inputs, &targets, cfg);
    }

    fn save_json(&self, path: String) -> PyResult<()> {
        self.model
            .save_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[staticmethod]
    fn load_json(path: String) -> PyResult<PyMLP> {
        let model = MLP::load_json(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PyMLP { model })
    }
}

#[pyfunction]
#[pyo3(signature = (model_type, params=None))]
fn create_model(
    py: Python<'_>,
    model_type: &str,
    params: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match model_type {
        "linear" | "linear_regression" => {
            let params =
                params.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing params"))?;
            let input_size: usize = params
                .get_item("input_size")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("input_size"))?
                .extract()?;
            Ok(Py::new(py, LinearRegression::new(input_size))?.into_any())
        }
        "mlp" => {
            let params =
                params.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing params"))?;
            let layer_sizes: Vec<usize> = params
                .get_item("layer_sizes")?
                .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("layer_sizes"))?
                .extract()?;
            Ok(Py::new(py, PyMLP::new(layer_sizes))?.into_any())
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(
            "unknown model_type",
        )),
    }
}

#[pyfunction]
fn train(
    model: &Bound<'_, PyAny>,
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    epochs: usize,
    lr: f64,
) -> PyResult<()> {
    if let Ok(mut m) = model.extract::<PyRefMut<'_, LinearRegression>>() {
        m.train(x, y, epochs, lr)
    } else if let Ok(mut m) = model.extract::<PyRefMut<'_, PyMLP>>() {
        m.train(x, y, lr, epochs);
        Ok(())
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("unsupported model"))
    }
}

#[pyfunction]
fn predict(model: &Bound<'_, PyAny>, x: Vec<f64>) -> PyResult<Vec<f64>> {
    if let Ok(m) = model.extract::<PyRef<'_, LinearRegression>>() {
        m.predict(x)
    } else if let Ok(m) = model.extract::<PyRef<'_, PyMLP>>() {
        Ok(m.predict(x))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("unsupported model"))
    }
}

#[pyfunction]
fn save(model: &Bound<'_, PyAny>, path: String) -> PyResult<()> {
    if let Ok(m) = model.extract::<PyRef<'_, LinearRegression>>() {
        let content = format!(
            "LinearRegression\ninput_size={}\nweights={:?}\nbias={}\n",
            m.model.input_size, m.model.weights, m.model.bias
        );
        fs::write(path, content).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "save is implemented only for LinearRegression",
        ))
    }
}

#[pyfunction]
fn load(py: Python<'_>, path: String) -> PyResult<Py<PyAny>> {
    let content = fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

    let mut input_size: Option<usize> = None;
    let mut weights: Option<Vec<f64>> = None;
    let mut bias: Option<f64> = None;

    for line in content.lines() {
        if let Some(v) = line.strip_prefix("input_size=") {
            input_size = Some(
                v.parse()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid input_size"))?,
            );
        } else if let Some(v) = line.strip_prefix("weights=") {
            let cleaned = v.trim().trim_start_matches('[').trim_end_matches(']');
            let parsed = if cleaned.is_empty() {
                Vec::new()
            } else {
                cleaned
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid weights"))?
            };
            weights = Some(parsed);
        } else if let Some(v) = line.strip_prefix("bias=") {
            bias = Some(
                v.parse()
                    .map_err(|_| pyo3::exceptions::PyValueError::new_err("invalid bias"))?,
            );
        }
    }

    let model = LinearRegression {
        model: LinearModel {
            input_size: input_size
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing input_size"))?,
            weights: weights
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing weights"))?,
            bias: bias.ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing bias"))?,
        },
    };

    Ok(Py::new(py, model)?.into_any())
}

#[pymodule]
fn vision_ai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVector>()?;
    m.add_class::<PyMatrix>()?;
    m.add_class::<LinearRegression>()?;
    m.add_class::<PyMLP>()?;
    m.add_class::<PyRBF>()?;
    m.add_class::<PySVM>()?;
    m.add_function(wrap_pyfunction!(create_model, m)?)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    m.add_function(wrap_pyfunction!(predict, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    Ok(())
}
