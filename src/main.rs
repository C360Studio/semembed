use anyhow::Context;
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use prometheus::{Counter, Encoder, Gauge, Histogram, HistogramOpts, Opts, Registry, TextEncoder};
use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    ops::{Not, Range},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};
use tokio::{net::TcpListener, sync::oneshot};
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::{error, info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const TOKENIZE_CHUNK_SIZE: usize = 256;
const DEFAULT_QUEUE_CAPACITY: usize = 128;
const DEFAULT_BATCH_WINDOW_MS: u64 = 5;
const DEFAULT_MIN_BATCH_TOKENS: usize = 4096;
const MAX_SUB_BATCH_ITEMS: usize = 64;
const MAX_WAVE_JOBS: usize = 64;
const ARCTIC_BGE_QUERY_PREFIX: &str =
    "Represent this sentence for searching relevant passages: ";
const SEARCH_QUERY_PREFIX: &str = "search_query: ";
const SEARCH_DOCUMENT_PREFIX: &str = "search_document: ";
const GEMMA_QUERY_PREFIX: &str = "task: search result | query: ";
const GEMMA_DOCUMENT_PREFIX: &str = "title: none | text: ";

#[derive(Clone)]
struct ModelSpec {
    model: EmbeddingModel,
    native_max_tokens: usize,
    query_prefix: Option<&'static str>,
    document_prefix: Option<&'static str>,
}

fn resolve_model(requested: &str) -> (String, ModelSpec) {
    let spec = match requested {
        "Snowflake/snowflake-arctic-embed-s" => ModelSpec {
            model: EmbeddingModel::SnowflakeArcticEmbedS,
            native_max_tokens: 512,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "Snowflake/snowflake-arctic-embed-xs" => ModelSpec {
            model: EmbeddingModel::SnowflakeArcticEmbedXS,
            native_max_tokens: 512,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "Snowflake/snowflake-arctic-embed-m" => ModelSpec {
            model: EmbeddingModel::SnowflakeArcticEmbedM,
            native_max_tokens: 512,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "Snowflake/snowflake-arctic-embed-m-long" => ModelSpec {
            model: EmbeddingModel::SnowflakeArcticEmbedMLong,
            native_max_tokens: 8192,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "nomic-ai/nomic-embed-text-v1.5" => ModelSpec {
            model: EmbeddingModel::NomicEmbedTextV15,
            native_max_tokens: 8192,
            query_prefix: Some(SEARCH_QUERY_PREFIX),
            document_prefix: Some(SEARCH_DOCUMENT_PREFIX),
        },
        "google/embeddinggemma-300m" | "onnx-community/embeddinggemma-300m-ONNX" => ModelSpec {
            model: EmbeddingModel::EmbeddingGemma300M,
            native_max_tokens: 2048,
            query_prefix: Some(GEMMA_QUERY_PREFIX),
            document_prefix: Some(GEMMA_DOCUMENT_PREFIX),
        },
        "google/embeddinggemma-300m-q" => ModelSpec {
            model: EmbeddingModel::EmbeddingGemma300MQ,
            native_max_tokens: 2048,
            query_prefix: Some(GEMMA_QUERY_PREFIX),
            document_prefix: Some(GEMMA_DOCUMENT_PREFIX),
        },
        "google/embeddinggemma-300m-q4" => ModelSpec {
            model: EmbeddingModel::EmbeddingGemma300MQ4,
            native_max_tokens: 2048,
            query_prefix: Some(GEMMA_QUERY_PREFIX),
            document_prefix: Some(GEMMA_DOCUMENT_PREFIX),
        },
        "lightonai/modernbert-embed-large" => ModelSpec {
            model: EmbeddingModel::ModernBertEmbedLarge,
            native_max_tokens: 8192,
            query_prefix: Some(SEARCH_QUERY_PREFIX),
            document_prefix: Some(SEARCH_DOCUMENT_PREFIX),
        },
        "BAAI/bge-small-en-v1.5" => ModelSpec {
            model: EmbeddingModel::BGESmallENV15,
            native_max_tokens: 512,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "BAAI/bge-base-en-v1.5" => ModelSpec {
            model: EmbeddingModel::BGEBaseENV15,
            native_max_tokens: 512,
            query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
            document_prefix: None,
        },
        "sentence-transformers/all-MiniLM-L6-v2" => ModelSpec {
            model: EmbeddingModel::AllMiniLML6V2,
            native_max_tokens: 256,
            query_prefix: None,
            document_prefix: None,
        },
        _ => {
            warn!(
                "Unknown model {}, defaulting to Snowflake/snowflake-arctic-embed-s",
                requested
            );
            return (
                "Snowflake/snowflake-arctic-embed-s".to_string(),
                ModelSpec {
                    model: EmbeddingModel::SnowflakeArcticEmbedS,
                    native_max_tokens: 512,
                    query_prefix: Some(ARCTIC_BGE_QUERY_PREFIX),
                    document_prefix: None,
                },
            );
        }
    };

    (requested.to_string(), spec)
}

fn resolve_max_tokens(model_name: &str, native_max_tokens: usize) -> anyhow::Result<usize> {
    match std::env::var("SEMEMBED_MAX_TOKENS") {
        Ok(raw) if !raw.trim().is_empty() => {
            let configured: usize = raw
                .trim()
                .parse()
                .with_context(|| format!("Invalid SEMEMBED_MAX_TOKENS value: {raw:?}"))?;
            anyhow::ensure!(configured > 0, "SEMEMBED_MAX_TOKENS must be greater than 0");
            if configured > native_max_tokens {
                warn!(
                    "SEMEMBED_MAX_TOKENS={} exceeds {} native window ({}); clamping to {}",
                    configured, model_name, native_max_tokens, native_max_tokens
                );
                Ok(native_max_tokens)
            } else {
                Ok(configured)
            }
        }
        _ => Ok(native_max_tokens),
    }
}

fn resolve_optional_positive_usize(var_name: &str) -> anyhow::Result<Option<usize>> {
    match std::env::var(var_name) {
        Ok(raw) if !raw.trim().is_empty() => {
            let value: usize = raw
                .trim()
                .parse()
                .with_context(|| format!("Invalid {var_name} value: {raw:?}"))?;
            anyhow::ensure!(value > 0, "{var_name} must be greater than 0");
            Ok(Some(value))
        }
        _ => Ok(None),
    }
}

fn resolve_batch_window() -> anyhow::Result<Duration> {
    match std::env::var("SEMEMBED_BATCH_WINDOW_MS") {
        Ok(raw) if !raw.trim().is_empty() => {
            let ms: u64 = raw
                .trim()
                .parse()
                .with_context(|| format!("Invalid SEMEMBED_BATCH_WINDOW_MS value: {raw:?}"))?;
            Ok(Duration::from_millis(ms))
        }
        _ => Ok(Duration::from_millis(DEFAULT_BATCH_WINDOW_MS)),
    }
}

fn has_known_prefix(text: &str, spec: &ModelSpec) -> bool {
    if spec.query_prefix.is_some_and(|p| text.starts_with(p))
        || spec.document_prefix.is_some_and(|p| text.starts_with(p))
    {
        return true;
    }
    // EmbeddingGemma's prompt format carries a variable field: callers may send their own
    // task ("task: question answering | query: ...") or a real title
    // ("title: Installation | text: ..."). Those are already prepared and must not be wrapped.
    spec.document_prefix == Some(GEMMA_DOCUMENT_PREFIX) && is_gemma_prepared(text)
}

fn is_gemma_prepared(text: &str) -> bool {
    let first_line = text.lines().next().unwrap_or_default();
    (text.starts_with("task: ") && first_line.contains(" | query: "))
        || (text.starts_with("title: ") && first_line.contains(" | text: "))
}

/// Partition ascending-sorted token lengths into sub-batch index ranges bounded by both
/// padded tensor size (`sub_batch_len * max_len_in_sub_batch <= max_batch_tokens`), item count,
/// and length spread (`<= 2x` once sequences exceed 64 tokens so short inputs are never padded
/// out to long documents). Always makes forward progress even when a single item exceeds
/// `max_batch_tokens`.
fn plan_sub_batches(
    sorted_token_lengths: &[usize],
    max_batch_tokens: usize,
    max_batch_items: usize,
) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let n = sorted_token_lengths.len();
    if n == 0 {
        return ranges;
    }

    let max_items = max_batch_items.max(1);
    let mut start = 0;
    while start < n {
        let first_len = sorted_token_lengths[start];
        let max_allowed_len = first_len.max(64).saturating_mul(2);
        let mut end = start + 1;

        while end < n && (end - start) < max_items {
            let next_len = sorted_token_lengths[end];
            let padded_tokens = (end - start + 1).saturating_mul(next_len);
            if next_len > max_allowed_len || padded_tokens > max_batch_tokens {
                break;
            }
            end += 1;
        }

        ranges.push(start..end);
        start = end;
    }

    ranges
}

// OpenAI-compatible request/response types
#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    input: InputType,
    #[allow(dead_code)]
    model: Option<String>,
    #[allow(dead_code)]
    #[serde(default)]
    encoding_format: EncodingFormat,
    #[serde(default)]
    input_type: Option<InputKind>,
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

#[derive(Debug, Clone, Copy, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum InputKind {
    Query,
    #[default]
    Document,
}

#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingObject>,
    model: String,
    usage: Usage,
    prefix_applied: bool,
}

#[derive(Debug, Serialize)]
struct EmbeddingObject {
    object: String,
    embedding: Vec<f32>,
    index: usize,
    #[serde(skip_serializing_if = "Not::not")]
    truncated: bool,
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
    max_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ModelsResponse {
    models: Vec<String>,
    details: Vec<ModelDetail>,
}

#[derive(Debug, Serialize)]
struct ModelDetail {
    id: String,
    max_tokens: usize,
    native_max_tokens: usize,
    query_prefix: Option<&'static str>,
    document_prefix: Option<&'static str>,
}

type BatchTokenizer = dyn Fn(&[String]) -> anyhow::Result<Vec<(usize, bool)>> + Send + Sync;
type JobReply = Result<Vec<Vec<f32>>, String>;

struct JobItem {
    text: String,
    token_len: usize,
}

struct QueuedJob {
    items: Vec<JobItem>,
    total_tokens: usize,
    reply: oneshot::Sender<JobReply>,
}

#[derive(Debug, PartialEq, Eq)]
enum EnqueueError {
    Full,
    Closed,
}

struct QueueInner {
    queries: VecDeque<QueuedJob>,
    documents: VecDeque<QueuedJob>,
    closed: bool,
}

struct InferenceQueue {
    inner: Mutex<QueueInner>,
    cv: Condvar,
    capacity: usize,
    batch_window: Duration,
    max_batch_tokens: usize,
    metrics: Arc<Metrics>,
}

impl InferenceQueue {
    fn new(
        capacity: usize,
        batch_window: Duration,
        max_batch_tokens: usize,
        metrics: Arc<Metrics>,
    ) -> Self {
        Self {
            inner: Mutex::new(QueueInner {
                queries: VecDeque::new(),
                documents: VecDeque::new(),
                closed: false,
            }),
            cv: Condvar::new(),
            capacity,
            batch_window,
            max_batch_tokens,
            metrics,
        }
    }

    fn max_wave_tokens(&self) -> usize {
        self.max_batch_tokens.saturating_mul(4)
    }

    fn is_alive(&self) -> bool {
        !self.inner.lock().unwrap().closed
    }

    fn mark_closed(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.closed = true;
        inner.queries.clear();
        inner.documents.clear();
        self.metrics.queue_depth.set(0.0);
        self.cv.notify_all();
    }

    fn try_enqueue(
        &self,
        kind: InputKind,
        items: Vec<JobItem>,
    ) -> Result<oneshot::Receiver<JobReply>, EnqueueError> {
        let total_tokens: usize = items.iter().map(|i| i.token_len).sum();
        let mut inner = self.inner.lock().unwrap();
        if inner.closed {
            return Err(EnqueueError::Closed);
        }

        // Reclaim slots from callers that disconnected while waiting.
        inner.queries.retain(|j| !j.reply.is_closed());
        inner.documents.retain(|j| !j.reply.is_closed());

        let depth = inner.queries.len() + inner.documents.len();
        if depth >= self.capacity {
            self.metrics.queue_depth.set(depth as f64);
            self.metrics.queue_rejected.inc();
            return Err(EnqueueError::Full);
        }

        let (tx, rx) = oneshot::channel();
        let job = QueuedJob {
            items,
            total_tokens,
            reply: tx,
        };
        match kind {
            InputKind::Query => inner.queries.push_back(job),
            InputKind::Document => inner.documents.push_back(job),
        }

        self.metrics.queue_depth.set((depth + 1) as f64);
        self.cv.notify_one();
        Ok(rx)
    }

    fn next_wave(&self) -> Option<(InputKind, Vec<QueuedJob>)> {
        let mut inner = self.inner.lock().unwrap();
        loop {
            inner.queries.retain(|j| !j.reply.is_closed());
            inner.documents.retain(|j| !j.reply.is_closed());
            self.metrics
                .queue_depth
                .set((inner.queries.len() + inner.documents.len()) as f64);

            if inner.queries.is_empty() && inner.documents.is_empty() {
                if inner.closed {
                    return None;
                }
                inner = self.cv.wait(inner).unwrap();
                continue;
            }

            if !self.batch_window.is_zero() {
                let deadline = Instant::now() + self.batch_window;
                loop {
                    let active = if !inner.queries.is_empty() {
                        &inner.queries
                    } else {
                        &inner.documents
                    };
                    let queued_tokens: usize = active.iter().map(|j| j.total_tokens).sum();
                    if queued_tokens >= self.max_batch_tokens
                        || active.len() >= MAX_WAVE_JOBS
                        || inner.closed
                    {
                        break;
                    }
                    let now = Instant::now();
                    if now >= deadline {
                        break;
                    }
                    let (guard, _) = self.cv.wait_timeout(inner, deadline - now).unwrap();
                    inner = guard;
                }

                inner.queries.retain(|j| !j.reply.is_closed());
                inner.documents.retain(|j| !j.reply.is_closed());
                if inner.queries.is_empty() && inner.documents.is_empty() {
                    continue;
                }
            }

            let max_wave_tokens = self.max_wave_tokens();
            let (kind, jobs) = if !inner.queries.is_empty() {
                (
                    InputKind::Query,
                    pop_from_deque(&mut inner.queries, max_wave_tokens),
                )
            } else {
                (
                    InputKind::Document,
                    pop_from_deque(&mut inner.documents, max_wave_tokens),
                )
            };

            self.metrics
                .queue_depth
                .set((inner.queries.len() + inner.documents.len()) as f64);
            return Some((kind, jobs));
        }
    }

    fn pop_ready_queries(&self) -> Option<Vec<QueuedJob>> {
        let mut inner = self.inner.lock().unwrap();
        inner.queries.retain(|j| !j.reply.is_closed());
        if inner.queries.is_empty() {
            self.metrics.queue_depth.set(inner.documents.len() as f64);
            return None;
        }
        let jobs = pop_from_deque(&mut inner.queries, self.max_wave_tokens());
        self.metrics
            .queue_depth
            .set((inner.queries.len() + inner.documents.len()) as f64);
        Some(jobs)
    }
}

fn pop_from_deque(deque: &mut VecDeque<QueuedJob>, max_wave_tokens: usize) -> Vec<QueuedJob> {
    let mut wave = Vec::new();
    let mut tokens = 0usize;
    while wave.len() < MAX_WAVE_JOBS {
        let Some(front) = deque.front() else {
            break;
        };
        if !wave.is_empty() && tokens.saturating_add(front.total_tokens) > max_wave_tokens {
            break;
        }
        let job = deque.pop_front().expect("checked front");
        tokens = tokens.saturating_add(job.total_tokens);
        wave.push(job);
    }
    wave
}

struct WorkerDropGuard {
    queue: Arc<InferenceQueue>,
}

impl Drop for WorkerDropGuard {
    fn drop(&mut self) {
        self.queue.mark_closed();
    }
}

struct FlatItem {
    job_slot: usize,
    item_index: usize,
    token_len: usize,
    text: String,
}

fn run_inference_worker(mut embedder: TextEmbedding, queue: Arc<InferenceQueue>) {
    let _guard = WorkerDropGuard {
        queue: Arc::clone(&queue),
    };
    while let Some((kind, jobs)) = queue.next_wave() {
        execute_wave(&mut embedder, kind, jobs, &queue);
    }
}

fn execute_wave(
    embedder: &mut TextEmbedding,
    kind: InputKind,
    jobs: Vec<QueuedJob>,
    queue: &InferenceQueue,
) {
    let mut active_jobs: Vec<(Vec<Option<Vec<f32>>>, oneshot::Sender<JobReply>)> =
        Vec::with_capacity(jobs.len());
    let mut flat_items: Vec<FlatItem> = Vec::new();

    for job in jobs {
        if job.reply.is_closed() {
            continue;
        }
        let job_slot = active_jobs.len();
        active_jobs.push((vec![None; job.items.len()], job.reply));
        for (item_index, item) in job.items.into_iter().enumerate() {
            flat_items.push(FlatItem {
                job_slot,
                item_index,
                token_len: item.token_len,
                text: item.text,
            });
        }
    }

    if flat_items.is_empty() {
        return;
    }

    flat_items.sort_by_key(|item| item.token_len);
    let sorted_lengths: Vec<usize> = flat_items.iter().map(|item| item.token_len).collect();
    let sub_batches = plan_sub_batches(&sorted_lengths, queue.max_batch_tokens, MAX_SUB_BATCH_ITEMS);

    for (sb_idx, range) in sub_batches.into_iter().enumerate() {
        // Between bulk document sub-batches, let waiting interactive search queries jump ahead
        // (at most one query wave per document sub-batch so document ingestion never starves).
        if kind == InputKind::Document && sb_idx > 0 {
            if let Some(query_jobs) = queue.pop_ready_queries() {
                execute_wave(embedder, InputKind::Query, query_jobs, queue);
            }
        }

        let refs: Vec<&str> = flat_items[range.clone()]
            .iter()
            .map(|item| item.text.as_str())
            .collect();
        queue.metrics.batch_size.observe(refs.len() as f64);
        let inf_timer = queue.metrics.inference_duration.start_timer();
        let result = embedder.embed(&refs, Some(refs.len()));
        inf_timer.observe_duration();

        match result {
            Ok(vectors) if vectors.len() == range.len() => {
                for (item, vec) in flat_items[range].iter().zip(vectors) {
                    active_jobs[item.job_slot].0[item.item_index] = Some(vec);
                }
            }
            Ok(vectors) => {
                let msg = format!(
                    "Embedding count mismatch: expected {}, got {}",
                    range.len(),
                    vectors.len()
                );
                error!("{}", msg);
                for (_, reply) in active_jobs {
                    let _ = reply.send(Err(msg.clone()));
                }
                return;
            }
            Err(e) => {
                let msg = format!("Failed to generate embeddings: {e}");
                error!("{}", msg);
                for (_, reply) in active_jobs {
                    let _ = reply.send(Err(msg.clone()));
                }
                return;
            }
        }
    }

    for (slots, reply) in active_jobs {
        let assembled: Result<Vec<Vec<f32>>, String> = slots
            .into_iter()
            .map(|opt| opt.ok_or_else(|| "Missing embedding result".to_string()))
            .collect();
        let _ = reply.send(assembled);
    }
}

// Application state
struct AppState {
    queue: Arc<InferenceQueue>,
    tokenize_batch: Arc<BatchTokenizer>,
    model_name: String,
    spec: ModelSpec,
    max_tokens: usize,
    metrics: Arc<Metrics>,
}

// Prometheus metrics
struct Metrics {
    registry: Registry,
    requests_total: Counter,
    request_duration: Histogram,
    inference_duration: Histogram,
    batch_size: Histogram,
    queue_depth: Gauge,
    queue_rejected: Counter,
    tokens_processed: Counter,
    inputs_truncated: Counter,
    errors_total: Counter,
}

impl Metrics {
    fn new() -> anyhow::Result<Self> {
        let registry = Registry::new();

        let requests_total = Counter::with_opts(Opts::new(
            "semembed_requests_total",
            "Total number of embedding requests",
        ))?;
        registry.register(Box::new(requests_total.clone()))?;

        let request_duration = Histogram::with_opts(HistogramOpts::new(
            "semembed_request_duration_seconds",
            "Request duration in seconds",
        ))?;
        registry.register(Box::new(request_duration.clone()))?;

        let inference_duration = Histogram::with_opts(
            HistogramOpts::new(
                "semembed_inference_duration_seconds",
                "ONNX sub-batch inference duration in seconds",
            )
            .buckets(vec![
                0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            ]),
        )?;
        registry.register(Box::new(inference_duration.clone()))?;

        let batch_size = Histogram::with_opts(
            HistogramOpts::new(
                "semembed_batch_size",
                "Number of inputs per ONNX inference sub-batch",
            )
            .buckets(vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]),
        )?;
        registry.register(Box::new(batch_size.clone()))?;

        let queue_depth = Gauge::with_opts(Opts::new(
            "semembed_queue_depth",
            "Current number of queued embedding requests waiting for inference",
        ))?;
        registry.register(Box::new(queue_depth.clone()))?;

        let queue_rejected = Counter::with_opts(Opts::new(
            "semembed_queue_rejected_total",
            "Total number of embedding requests rejected because the inference queue was full",
        ))?;
        registry.register(Box::new(queue_rejected.clone()))?;

        let tokens_processed = Counter::with_opts(Opts::new(
            "semembed_tokens_processed_total",
            "Total number of model tokens processed",
        ))?;
        registry.register(Box::new(tokens_processed.clone()))?;

        let inputs_truncated = Counter::with_opts(Opts::new(
            "semembed_inputs_truncated_total",
            "Total number of input texts truncated to the model context window",
        ))?;
        registry.register(Box::new(inputs_truncated.clone()))?;

        let errors_total = Counter::with_opts(Opts::new(
            "semembed_errors_total",
            "Total number of errors",
        ))?;
        registry.register(Box::new(errors_total.clone()))?;

        Ok(Self {
            registry,
            requests_total,
            request_duration,
            inference_duration,
            batch_size,
            queue_depth,
            queue_rejected,
            tokens_processed,
            inputs_truncated,
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
    let requested_model = std::env::var("SEMEMBED_MODEL")
        .unwrap_or_else(|_| "Snowflake/snowflake-arctic-embed-s".to_string());
    let port = std::env::var("SEMEMBED_PORT")
        .unwrap_or_else(|_| "8081".to_string())
        .parse::<u16>()
        .context("Invalid SEMEMBED_PORT value")?;
    let intra_threads = resolve_optional_positive_usize("SEMEMBED_INTRA_THREADS")?;
    let queue_capacity = resolve_optional_positive_usize("SEMEMBED_QUEUE_CAPACITY")?
        .unwrap_or(DEFAULT_QUEUE_CAPACITY);
    let batch_window = resolve_batch_window()?;

    let (model_name, spec) = resolve_model(&requested_model);
    let requested_max_tokens = resolve_max_tokens(&model_name, spec.native_max_tokens)?;

    info!(
        model = %model_name,
        max_tokens = requested_max_tokens,
        native_max_tokens = spec.native_max_tokens,
        intra_threads = ?intra_threads,
        "Loading embedding model"
    );

    // fastembed v7 API - TextInitOptions builder pattern with explicit model context limit
    let mut init_options = TextInitOptions::new(spec.model.clone())
        .with_max_length(requested_max_tokens)
        .with_show_download_progress(true);
    if let Some(threads) = intra_threads {
        init_options = init_options.with_intra_threads(threads);
    }
    let embedder = TextEmbedding::try_new(init_options)?;

    let active_max_tokens = embedder
        .tokenizer
        .get_truncation()
        .map(|t| t.max_length)
        .unwrap_or(requested_max_tokens);
    let max_batch_tokens = resolve_optional_positive_usize("SEMEMBED_MAX_BATCH_TOKENS")?
        .unwrap_or_else(|| active_max_tokens.max(DEFAULT_MIN_BATCH_TOKENS));

    // The tokenizer never cuts special tokens: a window that cannot hold them plus one content
    // token would silently drop text without reporting truncation, so refuse it at startup.
    let special_tokens: usize = embedder
        .tokenizer
        .encode("", true)
        .map_err(|e| anyhow::anyhow!("Failed to probe tokenizer special tokens: {e}"))?
        .get_attention_mask()
        .iter()
        .map(|&m| m as usize)
        .sum();
    anyhow::ensure!(
        active_max_tokens > special_tokens,
        "SEMEMBED_MAX_TOKENS={} is too small for {}: the model adds {} special tokens, so the minimum is {}",
        active_max_tokens,
        model_name,
        special_tokens,
        special_tokens + 1
    );

    // Clone the configured tokenizer so HTTP handlers can tokenize concurrently on blocking
    // pool threads without holding the ONNX session.
    let tokenizer = embedder.tokenizer.clone();
    let tokenize_batch: Arc<BatchTokenizer> = Arc::new(move |texts: &[String]| {
        let mut stats = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(TOKENIZE_CHUNK_SIZE) {
            let refs: Vec<&str> = chunk.iter().map(String::as_str).collect();
            let encodings = tokenizer
                .encode_batch(refs, true)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            for enc in encodings {
                let token_len: usize = enc.get_attention_mask().iter().map(|&m| m as usize).sum();
                let was_truncated = !enc.get_overflowing().is_empty();
                stats.push((token_len, was_truncated));
            }
        }
        Ok(stats)
    });

    // Initialize metrics and spawn dedicated ONNX inference worker thread
    let metrics = Arc::new(Metrics::new()?);
    let queue = Arc::new(InferenceQueue::new(
        queue_capacity,
        batch_window,
        max_batch_tokens,
        Arc::clone(&metrics),
    ));
    let worker_queue = Arc::clone(&queue);
    std::thread::Builder::new()
        .name("semembed-worker".to_string())
        .spawn(move || run_inference_worker(embedder, worker_queue))
        .context("Failed to spawn semembed-worker thread")?;

    info!(
        model = %model_name,
        max_tokens = active_max_tokens,
        native_max_tokens = spec.native_max_tokens,
        max_batch_tokens,
        queue_capacity,
        batch_window_ms = batch_window.as_millis() as u64,
        intra_threads = ?intra_threads,
        has_query_prefix = spec.query_prefix.is_some(),
        has_document_prefix = spec.document_prefix.is_some(),
        "Model loaded successfully"
    );

    // Create shared state
    let state = Arc::new(AppState {
        queue,
        tokenize_batch,
        model_name,
        spec,
        max_tokens: active_max_tokens,
        metrics,
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

    // Apply model-specific prefix and tokenize off the Tokio async worker pool before
    // entering the ONNX inference queue.
    let input_kind = req.input_type.unwrap_or_default();
    let target_prefix = match input_kind {
        InputKind::Query => state.spec.query_prefix,
        InputKind::Document => state.spec.document_prefix,
    };

    let state_for_tok = Arc::clone(&state);
    let tokenized = tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
        let mut prefix_applied = false;
        let prepared_texts: Vec<String> = texts
            .into_iter()
            .map(|text| match target_prefix {
                Some(prefix) if !has_known_prefix(&text, &state_for_tok.spec) => {
                    prefix_applied = true;
                    format!("{prefix}{text}")
                }
                _ => text,
            })
            .collect();

        let stats = (state_for_tok.tokenize_batch)(&prepared_texts)?;
        Ok((prepared_texts, stats, prefix_applied))
    })
    .await;

    let (prepared_texts, item_stats, prefix_applied) = match tokenized {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            error!("Failed to tokenize inputs: {}", e);
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: format!("Failed to tokenize inputs: {}", e),
                        error_type: "internal_error".to_string(),
                    },
                }),
            ));
        }
        Err(e) => {
            error!("Tokenizer task panicked: {}", e);
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Tokenizer task failed".to_string(),
                        error_type: "internal_error".to_string(),
                    },
                }),
            ));
        }
    };

    let job_items: Vec<JobItem> = prepared_texts
        .into_iter()
        .zip(&item_stats)
        .map(|(text, &(token_len, _))| JobItem { text, token_len })
        .collect();

    let rx = match state.queue.try_enqueue(input_kind, job_items) {
        Ok(rx) => rx,
        Err(EnqueueError::Full) => {
            warn!("Rejecting embedding request: inference queue is full");
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Inference queue is full; retry shortly".to_string(),
                        error_type: "rate_limit_error".to_string(),
                    },
                }),
            ));
        }
        Err(EnqueueError::Closed) => {
            error!("Rejecting embedding request: inference worker is not running");
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::SERVICE_UNAVAILABLE,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Inference worker is unavailable".to_string(),
                        error_type: "server_error".to_string(),
                    },
                }),
            ));
        }
    };

    let embeddings = match rx.await {
        Ok(Ok(emb)) => emb,
        Ok(Err(msg)) => {
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: msg,
                        error_type: "internal_error".to_string(),
                    },
                }),
            ));
        }
        Err(_) => {
            error!("Inference worker dropped response channel");
            state.metrics.errors_total.inc();
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse {
                    error: ErrorDetail {
                        message: "Inference worker terminated unexpectedly".to_string(),
                        error_type: "internal_error".to_string(),
                    },
                }),
            ));
        }
    };

    let prompt_tokens: usize = item_stats.iter().map(|(tokens, _)| *tokens).sum();
    let truncated_count = item_stats.iter().filter(|(_, truncated)| *truncated).count();

    state.metrics.tokens_processed.inc_by(prompt_tokens as f64);
    if truncated_count > 0 {
        state.metrics.inputs_truncated.inc_by(truncated_count as f64);
    }

    // Build response
    let data: Vec<EmbeddingObject> = embeddings
        .into_iter()
        .zip(item_stats)
        .enumerate()
        .map(|(index, (embedding, (_, truncated)))| EmbeddingObject {
            object: "embedding".to_string(),
            embedding,
            index,
            truncated,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: state.model_name.clone(),
        usage: Usage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
        prefix_applied,
    };

    timer.observe_duration();
    Ok(Json(response))
}

async fn health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.queue.is_alive() {
        (
            StatusCode::OK,
            Json(HealthResponse {
                status: "healthy".to_string(),
                model: state.model_name.clone(),
                max_tokens: state.max_tokens,
            }),
        )
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(HealthResponse {
                status: "unhealthy".to_string(),
                model: state.model_name.clone(),
                max_tokens: state.max_tokens,
            }),
        )
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(ModelsResponse {
        models: vec![state.model_name.clone()],
        details: vec![ModelDetail {
            id: state.model_name.clone(),
            max_tokens: state.max_tokens,
            native_max_tokens: state.spec.native_max_tokens,
            query_prefix: state.spec.query_prefix,
            document_prefix: state.spec.document_prefix,
        }],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_inputs_already_in_prompt_format_are_not_wrapped() {
        let (_, gemma) = resolve_model("google/embeddinggemma-300m");
        for prepared in [
            "title: none | text: Restart the worker",
            "title: Installation | text: Restart the worker",
            "task: search result | query: how do I restart the worker",
            "task: question answering | query: how do I restart the worker",
        ] {
            assert!(
                has_known_prefix(prepared, &gemma),
                "{prepared:?} should count as prepared"
            );
        }
        for plain in [
            "Restart the worker",
            "title: a heading in prose\nwith | text: on a later line",
            "the title: Installation | text: inline",
        ] {
            assert!(
                !has_known_prefix(plain, &gemma),
                "{plain:?} should get a prefix"
            );
        }
    }

    #[test]
    fn gemma_format_is_not_recognized_for_other_models() {
        let (_, arctic) = resolve_model("Snowflake/snowflake-arctic-embed-s");
        assert!(!has_known_prefix(
            "title: Installation | text: Restart",
            &arctic
        ));
        assert!(has_known_prefix(
            "Represent this sentence for searching relevant passages: restart",
            &arctic
        ));
    }

    #[test]
    fn sub_batch_planner_separates_short_and_long_sequences_and_bounds_padded_tokens() {
        let lengths = [10, 14, 32, 120, 1400, 1500, 8192];
        let ranges = plan_sub_batches(&lengths, 4096, 64);
        assert_eq!(ranges, vec![0..4, 4..6, 6..7]);
    }

    #[test]
    fn sub_batch_planner_always_advances_on_oversized_single_item() {
        let lengths = [8192, 8192];
        let ranges = plan_sub_batches(&lengths, 1024, 64);
        assert_eq!(ranges, vec![0..1, 1..2]);
    }

    #[test]
    fn queue_prioritizes_queries_enforces_capacity_and_reclaims_cancelled_jobs() {
        let metrics = Arc::new(Metrics::new().unwrap());
        let queue = InferenceQueue::new(2, Duration::ZERO, 4096, Arc::clone(&metrics));

        let doc_rx = queue
            .try_enqueue(
                InputKind::Document,
                vec![JobItem {
                    text: "doc".into(),
                    token_len: 100,
                }],
            )
            .unwrap();
        let query_rx = queue
            .try_enqueue(
                InputKind::Query,
                vec![JobItem {
                    text: "query".into(),
                    token_len: 12,
                }],
            )
            .unwrap();

        // Queue is at capacity (2): third enqueue must fail with Full and bump counter.
        let third = queue.try_enqueue(
            InputKind::Document,
            vec![JobItem {
                text: "overflow".into(),
                token_len: 10,
            }],
        );
        assert_eq!(third.unwrap_err(), EnqueueError::Full);
        assert_eq!(metrics.queue_rejected.get() as u64, 1);

        // Dropping a receiver frees its slot on the next enqueue attempt.
        drop(doc_rx);
        let doc2_rx = queue
            .try_enqueue(
                InputKind::Document,
                vec![JobItem {
                    text: "doc2".into(),
                    token_len: 80,
                }],
            )
            .unwrap();

        // Query wave must drain ahead of queued documents.
        let (first_kind, first_jobs) = queue.next_wave().unwrap();
        assert_eq!(first_kind, InputKind::Query);
        assert_eq!(first_jobs.len(), 1);
        assert_eq!(first_jobs[0].items[0].text, "query");
        drop(query_rx);

        let (second_kind, second_jobs) = queue.next_wave().unwrap();
        assert_eq!(second_kind, InputKind::Document);
        assert_eq!(second_jobs.len(), 1);
        assert_eq!(second_jobs[0].items[0].text, "doc2");
        drop(doc2_rx);
    }
}
