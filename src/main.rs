use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use prometheus::{Encoder, TextEncoder, Counter, Histogram, Registry, HistogramOpts, Opts};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{info, error, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

// OpenAI-compatible request/response types
#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    input: InputType,
    model: Option<String>,
    #[serde(default)]
    encoding_format: EncodingFormat,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum InputType {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
enum EncodingFormat {
    #[default]
    Float,
    Base64,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    #[serde(rename = "type")]
    error_type: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<String>,
}

// Application state
struct AppState {
    embedder: Mutex<TextEmbedding>,
    model_name: String,
    metrics: Arc<Metrics>,
}

// Prometheus metrics
struct Metrics {
    registry: Registry,
    requests_total: Counter,
    request_duration: Histogram,
    tokens_processed: Counter,
    errors_total: Counter,
}

impl Metrics {
    fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        let requests_total = Counter::with_opts(Opts::new(
            "semembed_requests_total",
            "Total number of embedding requests"
        ))?;
        registry.register(Box::new(requests_total.clone()))?;

        let request_duration = Histogram::with_opts(HistogramOpts::new(
            "semembed_request_duration_seconds",
            "Request duration in seconds"
        ))?;
        registry.register(Box::new(request_duration.clone()))?;

        let tokens_processed = Counter::with_opts(Opts::new(
            "semembed_tokens_processed_total",
            "Total number of tokens processed"
        ))?;
        registry.register(Box::new(tokens_processed.clone()))?;

        let errors_total = Counter::with_opts(Opts::new(
            "semembed_errors_total",
            "Total number of errors"
        ))?;
        registry.register(Box::new(errors_total.clone()))?;

        Ok(Self {
            registry,
            requests_total,
            request_duration,
            tokens_processed,
            errors_total,
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "semembed=info,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Starting semembed service");

    // Get configuration from environment
    let model_name = std::env::var("SEMEMBED_MODEL")
        .unwrap_or_else(|_| "BAAI/bge-small-en-v1.5".to_string());
    let port = std::env::var("SEMEMBED_PORT")
        .unwrap_or_else(|_| "8081".to_string())
        .parse::<u16>()?;

    info!("Loading embedding model: {}", model_name);

    // Initialize fastembed model
    let model = match model_name.as_str() {
        "BAAI/bge-small-en-v1.5" => EmbeddingModel::BGESmallENV15,
        "BAAI/bge-base-en-v1.5" => EmbeddingModel::BGEBaseENV15,
        "sentence-transformers/all-MiniLM-L6-v2" => EmbeddingModel::AllMiniLML6V2,
        _ => {
            warn!("Unknown model {}, defaulting to BGESmallENV15", model_name);
            EmbeddingModel::BGESmallENV15
        }
    };

    // fastembed v5 API - InitOptions builder pattern
    let embedder = TextEmbedding::try_new(
        InitOptions::new(model).with_show_download_progress(true)
    )?;

    info!("Model loaded successfully");

    // Initialize metrics
    let metrics = Arc::new(Metrics::new()?);

    // Create shared state
    let state = Arc::new(AppState {
        embedder: Mutex::new(embedder),
        model_name: model_name.clone(),
        metrics: metrics.clone(),
    });

    // Build router
    let app = Router::new()
        .route("/v1/embeddings", post(create_embeddings))
        .route("/health", get(health_check))
        .route("/models", get(list_models))
        .route("/metrics", get(metrics_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr = format!("0.0.0.0:{}", port);
    info!("Listening on {}", addr);

    let listener = TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn create_embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Json<EmbeddingResponse>, (StatusCode, Json<ErrorResponse>)> {
    let timer = state.metrics.request_duration.start_timer();
    state.metrics.requests_total.inc();

    // Extract texts from input
    let texts: Vec<String> = match req.input {
        InputType::Single(text) => vec![text],
        InputType::Batch(texts) => texts,
    };

    if texts.is_empty() {
        state.metrics.errors_total.inc();
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                error: ErrorDetail {
                    message: "Input cannot be empty".to_string(),
                    error_type: "invalid_request_error".to_string(),
                },
            }),
        ));
    }

    // Count tokens (approximate - count words for now)
    let token_count: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();
    state.metrics.tokens_processed.inc_by(token_count as f64);

    // Generate embeddings (lock the mutex for mutable access)
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = {
        let mut embedder = state.embedder.lock().unwrap();
        match embedder.embed(text_refs, None) {
            Ok(emb) => emb,
            Err(e) => {
                error!("Failed to generate embeddings: {}", e);
                state.metrics.errors_total.inc();
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(ErrorResponse {
                        error: ErrorDetail {
                            message: format!("Failed to generate embeddings: {}", e),
                            error_type: "internal_error".to_string(),
                        },
                    }),
                ));
            }
        }
    };

    // Build response
    let data: Vec<EmbeddingObject> = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingObject {
            object: "embedding".to_string(),
            embedding,
            index,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: state.model_name.clone(),
        usage: Usage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        },
    };

    timer.observe_duration();
    Ok(Json(response))
}

async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(HealthResponse {
        status: "healthy".to_string(),
        model: state.model_name.clone(),
    })
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelsResponse {
        models: vec![state.model_name.clone()],
    })
}

async fn metrics_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let encoder = TextEncoder::new();
    let metric_families = state.metrics.registry.gather();

    let mut buffer = Vec::new();
    if let Err(e) = encoder.encode(&metric_families, &mut buffer) {
        error!("Failed to encode metrics: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Failed to encode metrics".to_string(),
        );
    }

    match String::from_utf8(buffer) {
        Ok(metrics) => (StatusCode::OK, metrics),
        Err(e) => {
            error!("Failed to convert metrics to string: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                "Failed to convert metrics".to_string(),
            )
        }
    }
}
