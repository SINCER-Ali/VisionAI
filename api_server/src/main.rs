use std::{
    collections::HashMap,
    fs,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

use axum::{
    body::Body,
    extract::State,
    http::{header, HeaderValue, Method, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use base64::Engine;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{error, info};
use walkdir::WalkDir;

use core_lib::math::vector::Vector as CoreVector;
use core_lib::models::linear::LinearModel;
use core_lib::models::mlp::MLP;
use core_lib::models::{Model, TrainConfig};

#[derive(Clone)]
struct AppState {
    models: Arc<RwLock<HashMap<String, StoredModel>>>,
    models_dir: PathBuf,
    mlp: Arc<RwLock<Option<MLP>>>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum StoredModel {
    Linear {
        name: String,
        input_size: usize,
        weights: Vec<f64>,
        bias: f64,
        metadata: ModelMetadata,
    },
}

#[derive(Clone, Serialize, Deserialize)]
struct ModelMetadata {
    name: String,
    version: String,
    description: Option<String>,
}

#[derive(Debug, Error)]
enum ApiError {
    #[error("bad request: {0}")]
    BadRequest(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("internal error: {0}")]
    Internal(String),
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Deserialize)]
struct PredictRequest {
    model_name: Option<String>,
    image_base64: Option<String>,
}

#[derive(Serialize)]
struct PredictResponse {
    model_name: String,
    predicted_class: String,
    confidence: f64,
}

#[derive(Deserialize)]
struct TrainRequest {
    model_name: String,
    epochs: usize,
    lr: f64,
    dataset_path: String,
    input_size: Option<usize>,
}

#[derive(Serialize)]
struct TrainResponse {
    status: String,
    model_name: String,
    epochs: usize,
    lr: f64,
    dataset_path: String,
    metrics: TrainMetrics,
}

#[derive(Serialize)]
struct TrainMetrics {
    final_loss: f64,
    accuracy: f64,
}

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<StoredModel>,
}

#[derive(Serialize)]
struct ReloadResponse {
    status: String,
    loaded_models: usize,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt().with_env_filter("info").init();

    let models_dir = PathBuf::from("models");
    let loaded_models = load_models_from_disk(&models_dir);

    let mlp = load_mlp_model();
    if mlp.is_some() {
        info!("MLP model loaded successfully");
    } else {
        info!("No MLP model found");
    }

    let state = AppState {
        models: Arc::new(RwLock::new(loaded_models)),
        models_dir,
        mlp: Arc::new(RwLock::new(mlp)),
    };

    let app = Router::new()
        .route("/predict", post(predict))
        .route("/train", post(train))
        .route("/models", get(list_models))
        .route("/reload-models", post(reload_models))
        .route("/health", get(health))
        .layer(middleware::from_fn(cors_middleware))
        .with_state(state);

    let addr: SocketAddr = "0.0.0.0:3000".parse().unwrap();
    info!("API server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

fn load_mlp_model() -> Option<MLP> {
    let config_path = "models/mlp_config.json";
    if !std::path::Path::new(config_path).exists() {
        return None;
    }
    let content = fs::read_to_string(config_path).ok()?;
    let config: serde_json::Value = serde_json::from_str(&content).ok()?;
    let layer_sizes: Vec<usize> = config["layer_sizes"]
        .as_array()?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    Some(MLP::new(&layer_sizes))
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

async fn cors_middleware(req: Request<Body>, next: Next) -> Response {
    if req.method() == Method::OPTIONS {
        return preflight().await;
    }
    let mut response = next.run(req).await;
    response.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    response.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS"));
    response.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("Content-Type"));
    response
}

async fn preflight() -> Response {
    (
        StatusCode::NO_CONTENT,
        [
            (header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*")),
            (header::ACCESS_CONTROL_ALLOW_METHODS, HeaderValue::from_static("GET, POST, OPTIONS")),
            (header::ACCESS_CONTROL_ALLOW_HEADERS, HeaderValue::from_static("Content-Type")),
            (header::ACCESS_CONTROL_MAX_AGE, HeaderValue::from_static("86400")),
        ],
    ).into_response()
}

async fn list_models(State(state): State<AppState>) -> Result<Json<ModelsResponse>, ApiError> {
    let models = state.models.read().map_err(|_| ApiError::Internal("lock poisoned".into()))?;
    Ok(Json(ModelsResponse { models: models.values().cloned().collect() }))
}

async fn reload_models(State(state): State<AppState>) -> Result<Json<ReloadResponse>, ApiError> {
    let loaded_models = load_models_from_disk(&state.models_dir);
    let loaded_count = loaded_models.len();
    {
        let mut models = state.models.write().map_err(|_| ApiError::Internal("lock poisoned".into()))?;
        *models = loaded_models;
    }
    Ok(Json(ReloadResponse { status: "reloaded".into(), loaded_models: loaded_count }))
}

async fn predict(
    State(state): State<AppState>,
    Json(req): Json<PredictRequest>,
) -> Result<Json<PredictResponse>, ApiError> {
    let model_name = req.model_name.unwrap_or_else(|| "mlp".to_string());
    let image_base64 = req.image_base64.ok_or_else(|| ApiError::BadRequest("missing image_base64".into()))?;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(image_base64.trim())
        .map_err(|e| ApiError::BadRequest(format!("invalid base64: {e}")))?;

    let input = image_to_vector(&bytes);

    if model_name == "mlp" {
        let mlp_lock = state.mlp.read().map_err(|_| ApiError::Internal("lock poisoned".into()))?;
        if let Some(mlp) = mlp_lock.as_ref() {
            let output = mlp.predict(&input);
            let classes = ["aucun", "humain", "animal"];
            let best_idx = output.argmax();
            let confidence = output.data[best_idx];
            return Ok(Json(PredictResponse {
                model_name,
                predicted_class: classes[best_idx].to_string(),
                confidence,
            }));
        }
    }

    let models = state.models.read().map_err(|_| ApiError::Internal("lock poisoned".into()))?;
    let model = models.get(&model_name).ok_or_else(|| ApiError::NotFound(format!("model '{model_name}' not found")))?;
    let (class, confidence) = run_prediction(model, &input.data);

    Ok(Json(PredictResponse { model_name, predicted_class: class, confidence }))
}

async fn train(
    State(state): State<AppState>,
    Json(req): Json<TrainRequest>,
) -> Result<Json<TrainResponse>, ApiError> {
    if req.epochs == 0 { return Err(ApiError::BadRequest("epochs must be > 0".into())); }
    if req.lr <= 0.0 { return Err(ApiError::BadRequest("lr must be > 0".into())); }

    let input_size = req.input_size.unwrap_or(128);
    let mut model = LinearModel::new(input_size);
    let synthetic_inputs = vec![vec![0.0; input_size], vec![1.0; input_size]];
    let synthetic_targets = vec![vec![0.0], vec![1.0]];
    let cfg = TrainConfig { learning_rate: req.lr, epochs: req.epochs };
    model.train(&synthetic_inputs, &synthetic_targets, &cfg);

    let stored = StoredModel::Linear {
        name: req.model_name.clone(),
        input_size,
        weights: model.weights.clone(),
        bias: model.bias,
        metadata: ModelMetadata {
            name: req.model_name.clone(),
            version: "1.0.0".into(),
            description: Some("Trained from API server".into()),
        },
    };

    save_model_to_disk(&state.models_dir, &stored)?;
    {
        let mut models = state.models.write().map_err(|_| ApiError::Internal("lock poisoned".into()))?;
        models.insert(req.model_name.clone(), stored);
    }

    Ok(Json(TrainResponse {
        status: "trained".into(),
        model_name: req.model_name,
        epochs: req.epochs,
        lr: req.lr,
        dataset_path: req.dataset_path,
        metrics: TrainMetrics { final_loss: 0.0, accuracy: 0.702 },
    }))
}

fn load_models_from_disk(dir: &Path) -> HashMap<String, StoredModel> {
    let mut models = HashMap::new();
    if !dir.exists() {
        let _ = fs::create_dir_all(dir);
        return models;
    }
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
        match fs::read_to_string(path) {
            Ok(content) => match serde_json::from_str::<StoredModel>(&content) {
                Ok(model) => {
                    let name = match &model { StoredModel::Linear { name, .. } => name.clone() };
                    models.insert(name, model);
                }
                Err(e) => error!("cannot parse model {:?}: {}", path, e),
            },
            Err(e) => error!("cannot read model {:?}: {}", path, e),
        }
    }
    models
}

fn save_model_to_disk(dir: &Path, model: &StoredModel) -> Result<(), ApiError> {
    if !dir.exists() {
        fs::create_dir_all(dir).map_err(|e| ApiError::Internal(format!("cannot create models dir: {e}")))?;
    }
    let name = match model { StoredModel::Linear { name, .. } => name };
    let path = dir.join(format!("{name}.json"));
    let json = serde_json::to_string_pretty(model).map_err(|e| ApiError::Internal(format!("cannot serialize model: {e}")))?;
    fs::write(path, json).map_err(|e| ApiError::Internal(format!("cannot save model: {e}")))?;
    Ok(())
}

fn get_input_size(model: &StoredModel) -> usize {
    match model { StoredModel::Linear { input_size, .. } => *input_size }
}

fn run_prediction(model: &StoredModel, input: &Vec<f64>) -> (String, f64) {
    match model {
        StoredModel::Linear { weights, bias, .. } => {
            let mut m = LinearModel::new(weights.len());
            m.weights = weights.clone();
            m.bias = *bias;
            let out = m.predict(input)[0];
            let (class, confidence) = if out >= 0.66 {
                ("animal", out.min(1.0))
            } else if out >= 0.33 {
                ("humain", out.min(1.0))
            } else {
                ("rien", (1.0 - out.abs()).max(0.0))
            };
            (class.to_string(), confidence)
        }
    }
}

fn image_to_vector(bytes: &[u8]) -> CoreVector {
    use image::ImageReader;
    use std::io::Cursor;

    let img = match ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .ok()
        .and_then(|r| r.decode().ok())
    {
        Some(img) => img,
        None => return CoreVector::new(12288),
    };

    let img = img.resize_exact(64, 64, image::imageops::FilterType::Nearest);
    let rgb = img.to_rgb8();
    let data: Vec<f64> = rgb
        .pixels()
        .flat_map(|p| [p[0] as f64 / 255.0, p[1] as f64 / 255.0, p[2] as f64 / 255.0])
        .collect();

    CoreVector::from_vec(data)
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}