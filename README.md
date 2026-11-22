# semembed

Lightweight HTTP embedding service for SemStreams using [fastembed-rs](https://github.com/Anush008/fastembed-rs).

## Overview

`semembed` provides text embeddings via an OpenAI-compatible HTTP API (`/v1/embeddings`).
**Key Benefits**:

- Multi-architecture support (linux/amd64, linux/arm64) without emulation
- Automatic model downloading and caching
- OpenAI-compatible API for drop-in replacement
- Smaller memory footprint than TEI (~512MB vs 1-2GB)
- Prometheus metrics for observability

## Containerized Development

**No local Rust required!** All development uses Docker to avoid toolchain setup.

**Quick Start**:

```bash
# Build and run service
task dev

# View logs
task logs

# Test embeddings
task test:embed

# Clean up
task clean
```

See [QUICKSTART.md](./QUICKSTART.md) for 5-minute getting started guide.
See [Taskfile.yml](./Taskfile.yml) for all available tasks.

## Quick Start

### Using Docker Compose

```bash
# Start with default model (Snowflake/snowflake-arctic-embed-s)
docker compose -f docker-compose.services.yml --profile embedding up -d

# Check health
curl http://localhost:8081/health

# Generate embeddings
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "Snowflake/snowflake-arctic-embed-s"
  }'
```

### Image tags

Images are published to `ghcr.io/c360studio/semembed`:

| Tag | Published on | Use |
|-----|--------------|-----|
| `X.Y.Z` (e.g. `0.2.0`) | a `vX.Y.Z` release | Pin this in deployments |
| `X.Y` (e.g. `0.2`) | the latest `X.Y.*` release | Pick up patch fixes |
| `main-<sha>` | every push to `main` | A specific unreleased commit |
| `latest` | every push to `main` | Local trials only; it changes without notice |

Releases follow semver, with 0.x until the API settles: a minor bump
(`0.x.0`) can change behavior or embedding vectors, a patch bump
(`0.x.y`) does not. To cut one: `gh release create vX.Y.Z --generate-notes`.

### Building from Source

```bash
# Build Docker image
docker build -t semstreams-semembed:latest .

# Run container
docker run -p 8081:8081 \
  -e SEMEMBED_MODEL=Snowflake/snowflake-arctic-embed-s \
  -e RUST_LOG=info \
  semstreams-semembed:latest
```

## API Reference

### POST /v1/embeddings

OpenAI-compatible embedding generation endpoint.

**Request**:

```json
{
  "input": "Text to embed",
  "model": "Snowflake/snowflake-arctic-embed-s",
  "encoding_format": "float",
  "input_type": "document"
}
```

**Input field types**:

- Single string: `"input": "text"`
- Array of strings: `"input": ["text1", "text2"]`

**`input_type`** (optional): `"query"` or `"document"`, default `"document"`.
semembed adds the served model's own prefix so clients don't have to know it:

| Model family | `query` prefix | `document` prefix |
|--------------|----------------|-------------------|
| `Snowflake/snowflake-arctic-embed-*`, `BAAI/bge-*-en-v1.5` | `Represent this sentence for searching relevant passages: ` | none |
| `nomic-ai/nomic-embed-text-v1.5`, `lightonai/modernbert-embed-large` | `search_query: ` | `search_document: ` |
| `google/embeddinggemma-300m*` | `task: search result \| query: ` | `title: none \| text: ` |
| `sentence-transformers/all-MiniLM-L6-v2` | none | none |

A text that already starts with one of the model's known prefixes is sent
unchanged, so clients that still add their own prefix are not
double-prefixed. For EmbeddingGemma, any input already in its prompt
format counts, including a custom task (`task: question answering | query: ...`)
or a real title (`title: Installation | text: ...`). `GET /models` lists the
prefixes in use.

> For the nomic, ModernBERT and EmbeddingGemma models, a request without
> `input_type` now gets the document prefix. Requests to the default model
> without `input_type` are unchanged.

**Context window**: each text is cut to the served model's context window
(see `max_tokens` in `GET /health`), or to `SEMEMBED_MAX_TOKENS` if that is
lower. A text that was cut comes back with `"truncated": true` on its `data`
entry and is counted in `semembed_inputs_truncated_total`.

**Response**:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "Snowflake/snowflake-arctic-embed-s",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  },
  "prefix_applied": false
}
```

- `truncated: true` appears only on entries that were cut; it is omitted
  otherwise. Clients that follow the OpenAI shape ignore it.
- `usage.prompt_tokens` is the real model token count from the model's
  tokenizer, including special tokens and any added prefix, capped at the
  context window for cut texts.
- `prefix_applied` is `true` when semembed added a prefix to at least one
  input.

### GET /health

Health check endpoint for container orchestration.

**Response**:

```json
{
  "status": "healthy",
  "model": "Snowflake/snowflake-arctic-embed-s",
  "max_tokens": 512
}
```

### GET /models

List loaded models endpoint.

**Response**:

```json
{
  "models": ["Snowflake/snowflake-arctic-embed-s"],
  "details": [
    {
      "id": "Snowflake/snowflake-arctic-embed-s",
      "max_tokens": 512,
      "native_max_tokens": 512,
      "query_prefix": "Represent this sentence for searching relevant passages: ",
      "document_prefix": null
    }
  ]
}
```

### GET /metrics

Prometheus metrics endpoint.

**Metrics**:

- `semembed_requests_total` - Total embedding requests
- `semembed_request_duration_seconds` - End-to-end HTTP request latency histogram
- `semembed_inference_duration_seconds` - ONNX sub-batch inference duration histogram
- `semembed_batch_size` - Inputs per ONNX inference sub-batch histogram
- `semembed_queue_depth` - Current number of requests waiting in the inference queue
- `semembed_queue_rejected_total` - Requests rejected with HTTP 429 because the inference queue was full
- `semembed_tokens_processed_total` - Total model tokens processed
- `semembed_inputs_truncated_total` - Input texts cut to the context window
- `semembed_errors_total` - Total errors

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMEMBED_MODEL` | `Snowflake/snowflake-arctic-embed-s` | Model to use (see supported models) |
| `SEMEMBED_MAX_TOKENS` | model's own window | Lower the context window to bound memory and latency (clamped to the model's window if set higher; startup fails below the model's special-token count + 1, e.g. 3 for the BERT-style models) |
| `SEMEMBED_MAX_BATCH_TOKENS` | `max(4096, max_tokens)` | Maximum padded token budget (`items × max_seq_len`) per ONNX sub-batch; items are sorted by token length so short inputs are not padded out to long documents |
| `SEMEMBED_BATCH_WINDOW_MS` | `5` | Coalescing window (ms) for dynamic micro-batching across concurrent requests (`0` disables waiting; `input_type=query` waves always take priority over `document` ingestion) |
| `SEMEMBED_QUEUE_CAPACITY` | `128` | Maximum queued requests before returning `429 Too Many Requests` (`rate_limit_error`) |
| `SEMEMBED_INTRA_THREADS` | all available CPU cores | Cap ONNX Runtime intra-op CPU threads (recommended when running with Docker CPU limits) |
| `SEMEMBED_PORT` | `8081` | HTTP server port |
| `RUST_LOG` | `info` | Log level (error, warn, info, debug, trace) |

## Supported Models

Models are automatically downloaded by fastembed-rs on first startup. The
default (`Snowflake/snowflake-arctic-embed-s`) is the strongest option in
the small/fast tier per the MTEB Retrieval leaderboard (NDCG@10 51.98 vs
51.68 for `bge-small-en-v1.5`); same 33M params, same 384-dim, same 512
token context, Apache-2.0.

The Context column is the token window semembed applies (it is passed to
fastembed explicitly; fastembed's own default would cut every model at 512).

| Model | Dimensions | Params | Context | Best For |
|-------|------------|--------|---------|----------|
| `Snowflake/snowflake-arctic-embed-s` *(default)* | 384 | 33M | 512 | Best small-tier 384d retrieval quality |
| `Snowflake/snowflake-arctic-embed-xs` | 384 | 22M | 512 | Smallest/fastest; tightest budgets |
| `Snowflake/snowflake-arctic-embed-m` | 768 | 110M | 512 | Higher quality, ~3× the cost |
| `Snowflake/snowflake-arctic-embed-m-long` | 768 | 137M | 8192 | Long-context retrieval (NDCG@10 54.83) |
| `nomic-ai/nomic-embed-text-v1.5` | 768 | 137M | 8192 | Long-context with Matryoshka support |
| `google/embeddinggemma-300m` | 768 | 300M | 2048 | Multilingual (100+ langs) & code retrieval |
| `google/embeddinggemma-300m-q` | 768 | 300M | 2048 | Quantized EmbeddingGemma for <1GB RAM |
| `google/embeddinggemma-300m-q4` | 768 | 300M | 2048 | 4-bit quantized EmbeddingGemma |
| `lightonai/modernbert-embed-large` | 1024 | 395M | 8192 | High overall MTEB (63.84) with 8K context |
| `BAAI/bge-small-en-v1.5` | 384 | 33M | 512 | Prior default, kept for compat |
| `BAAI/bge-base-en-v1.5` | 768 | 109M | 512 | Prior higher-quality option |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22M | 256 | Baseline; outperformed by arctic-xs |

> **Vector compatibility**: switching models produces incompatible
> embedding spaces. Vectors from one model cannot be compared with
> vectors from another. If you have persisted embeddings, re-embed them
> after changing the model.

To change models, set `SEMEMBED_MODEL` environment variable:

```bash
docker run -p 8081:8081 \
  -e SEMEMBED_MODEL=Snowflake/snowflake-arctic-embed-m \
  semstreams-semembed:latest
```

### Long context: memory and latency

A long window costs memory that grows faster than the token count
(attention is quadratic in sequence length), and ONNX Runtime keeps the
peak allocation after the request ends. One 8,192-token text needs far more
than the 512 MB to 1 GB a small container is usually given, so set
`SEMEMBED_MAX_TOKENS` for long-window models when memory is tight.

Measured with `Snowflake/snowflake-arctic-embed-m-long`: one fresh
container per window, so each peak is from that row's request alone. Docker
Desktop on an Apple-silicon Mac (arm64, 12 CPUs); x86 numbers will differ.
Reproduce with `task bench:context`.

| `max_tokens` | Inputs per request | Latency | Peak RSS |
|-------------:|-------------------:|--------:|---------:|
| (idle, model loaded) | – | – | 654 MiB |
| 512 | 1 | 0.3 s | 659 MiB |
| 512 | 4 | 1.0 s | 806 MiB |
| 1,024 | 1 | 0.6 s | 782 MiB |
| 1,024 | 4 | 2.4 s | 1,271 MiB |
| 2,048 | 1 | 1.4 s | 1,130 MiB |
| 2,048 | 4 | 6.3 s | 2,751 MiB |
| 4,096 | 1 | 4.0 s | 2,441 MiB |
| 8,192 | 1 | 10.7 s | 7,433 MiB |

Peaks add up across requests of different shapes: one container that
served a mix of 1,700- to 8,192-token requests reached 18.8 GiB. Batches of
8,192-token inputs were not measured.

## Architecture

```text
┌─────────────────────────────────┐
│   SemStreams Graph Processor    │
│   - HTTP Embedding Client       │
│   - BM25 Fallback (optional)    │
└────────────┬────────────────────┘
             │ HTTP POST /v1/embeddings
             ↓
┌─────────────────────────────────┐
│   semembed HTTP Server (Rust)   │
│   - Axum web framework          │
│   - OpenAI-compatible API       │
│   - Prometheus metrics          │
└────────────┬────────────────────┘
             │
             ↓
┌─────────────────────────────────┐
│   fastembed-rs                  │
│   - Model downloading           │
│   - ONNX Runtime                │
│   - Tokenization                │
└─────────────────────────────────┘
```

## Integration with SemStreams

The graph processor's indexmanager can use semembed for semantic search:

```bash
# Environment variables for semstreams
EMBEDDING_PROVIDER=http
EMBEDDING_HTTP_ENDPOINT=http://semembed:8081/v1/embeddings
EMBEDDING_HTTP_MODEL=Snowflake/snowflake-arctic-embed-s
```

See `processor/graph/indexmanager/embedding/http_embedder.go` for implementation.

## Development Workflow (Task-based)

All development tasks use Docker - **no local Rust installation required**.

### Quick Commands

```bash
# Full development cycle (build + run + test)
task dev

# Build Docker image
task build

# Run service (background)
task run

# Run service (foreground, see logs)
task run:fg

# Run with different models
task run:arctic-xs      # snowflake-arctic-embed-xs (22M, smallest/fastest)
task run:arctic-m       # snowflake-arctic-embed-m (110M, higher quality)
task run:bge-small      # bge-small-en-v1.5 (prior default)

# Test endpoints
task test:health
task test:embed         # Single embedding
task test:batch         # Batch embeddings
task test:all

# Monitor
task logs               # Follow logs
task metrics            # View Prometheus metrics

# Development
task restart            # Restart service
task shell              # Open container shell
task dev:rebuild        # Clean + rebuild + run

# Cleanup
task clean
task clean:all
```

### Docker Compose Workflow

```bash
# Start service
docker compose up -d

# View logs
docker compose logs -f

# Stop service
docker compose down
```

Or use Task shortcuts:

```bash
task compose:up
task compose:logs
task compose:down
```

### Native Rust Development (Optional)

For direct Rust development without Docker:

**Running Locally**:

```bash
# Install dependencies
cargo build

# Run with default settings
cargo run

# Run with custom model
SEMEMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2 cargo run

# Run with debug logging
RUST_LOG=debug cargo run
```

**Testing**:

```bash
# Run tests
cargo test

# Test the API
curl -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": ["test text"], "model": "Snowflake/snowflake-arctic-embed-s"}'
```

### Building Multi-Arch Images

```bash
# Build for current platform
docker build -t semembed:latest .

# Build for multiple platforms (requires buildx)
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t semembed:latest \
  --push .
```

## Resource Requirements

**Minimum**:

- 512MB RAM
- 1 CPU core
- 500MB disk space (model cache)

**Recommended**:

- 1GB RAM
- 2 CPU cores
- 1GB disk space

## Troubleshooting

### Model Download Fails

Models are downloaded on first startup. Ensure:

- Container has internet access
- Sufficient disk space for model cache
- `~/.cache/fastembed` directory is writable

### Out of Memory

Reduce memory usage by:

- Using smaller model (all-MiniLM-L6-v2)
- Limiting container memory: `docker run -m 512M`
- Reducing batch sizes in client

### Slow Performance

Improve performance by:

- Allocating more CPU cores
- Using larger model on powerful hardware
- Batching requests when possible

## CI/CD

GitHub Actions workflow automatically:

- Builds Docker image
- Runs health checks
- Tests all endpoints (single, batch embeddings)

**Workflow**: `.github/workflows/ci.yml`

**Run CI locally**:

```bash
task ci:test
```

## License

MIT

## Related

- [Taskfile.yml](./Taskfile.yml) - Development automation tasks
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) - Rust embedding library
- [SemStreams](https://github.com/C360Studio/semstreams) - Core framework
- [Taskfile Documentation](https://taskfile.dev) - Task runner docs
