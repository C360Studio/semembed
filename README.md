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
  "encoding_format": "float"
}
```

**Input field types**:

- Single string: `"input": "text"`
- Array of strings: `"input": ["text1", "text2"]`

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
  }
}
```

### GET /health

Health check endpoint for container orchestration.

**Response**:

```json
{
  "status": "healthy",
  "model": "Snowflake/snowflake-arctic-embed-s"
}
```

### GET /models

List loaded models endpoint.

**Response**:

```json
{
  "models": ["Snowflake/snowflake-arctic-embed-s"]
}
```

### GET /metrics

Prometheus metrics endpoint.

**Metrics**:

- `semembed_requests_total` - Total embedding requests
- `semembed_request_duration_seconds` - Request latency histogram
- `semembed_tokens_processed_total` - Total tokens processed
- `semembed_errors_total` - Total errors

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SEMEMBED_MODEL` | `Snowflake/snowflake-arctic-embed-s` | Model to use (see supported models) |
| `SEMEMBED_PORT` | `8081` | HTTP server port |
| `RUST_LOG` | `info` | Log level (error, warn, info, debug, trace) |

## Supported Models

Models are automatically downloaded by fastembed-rs on first startup. The
default (`Snowflake/snowflake-arctic-embed-s`) is the strongest option in
the small/fast tier per the MTEB Retrieval leaderboard (NDCG@10 51.98 vs
51.68 for `bge-small-en-v1.5`); same 33M params, same 384-dim, same 512
token context, Apache-2.0.

| Model | Dimensions | Params | Best For |
|-------|------------|--------|----------|
| `Snowflake/snowflake-arctic-embed-s` *(default)* | 384 | 33M | Best small-tier retrieval quality |
| `Snowflake/snowflake-arctic-embed-xs` | 384 | 22M | Smallest/fastest; tightest budgets |
| `Snowflake/snowflake-arctic-embed-m` | 768 | 110M | Higher quality, ~3× the cost |
| `BAAI/bge-small-en-v1.5` | 384 | 33M | Prior default, kept for compat |
| `BAAI/bge-base-en-v1.5` | 768 | 109M | Prior higher-quality option |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 22M | Baseline; outperformed by arctic-xs |

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
