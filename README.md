# semembed

Lightweight HTTP embedding service for SemStreams using [fastembed-rs](https://github.com/Anush008/fastembed-rs).

**📚 Ecosystem Documentation**: For SemStreams architecture, integration guides, and deployment strategies, see [semdocs](https://github.com/c360/semdocs). This README covers semembed implementation details.

## Overview

`semembed` provides text embeddings via an OpenAI-compatible HTTP API (`/v1/embeddings`). It replaces the previous Go+ONNX implementation with a simpler Rust service that handles all model management, tokenization, and ONNX Runtime complexity internally.

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
# Start with default model (BAAI/bge-small-en-v1.5)
docker compose -f docker-compose.services.yml --profile embedding up -d

# Check health
curl http://localhost:8081/health

# Generate embeddings
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world",
    "model": "BAAI/bge-small-en-v1.5"
  }'
```

### Building from Source

```bash
# Build Docker image
docker build -t semstreams-semembed:latest .

# Run container
docker run -p 8081:8081 \
  -e SEMEMBED_MODEL=BAAI/bge-small-en-v1.5 \
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
  "model": "BAAI/bge-small-en-v1.5",
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
  "model": "BAAI/bge-small-en-v1.5",
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
  "model": "BAAI/bge-small-en-v1.5"
}
```

### GET /models

List loaded models endpoint.

**Response**:
```json
{
  "models": ["BAAI/bge-small-en-v1.5"]
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
| `SEMEMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Model to use (see supported models) |
| `SEMEMBED_PORT` | `8081` | HTTP server port |
| `RUST_LOG` | `info` | Log level (error, warn, info, debug, trace) |

## Supported Models

Models are automatically downloaded by fastembed-rs on first startup:

| Model | Dimensions | Size | Best For |
|-------|------------|------|----------|
| `BAAI/bge-small-en-v1.5` | 384 | ~120MB | General purpose, fast |
| `BAAI/bge-base-en-v1.5` | 768 | ~420MB | Higher quality |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | ~90MB | Fast, good quality |

To change models, set `SEMEMBED_MODEL` environment variable:

```bash
docker run -p 8081:8081 \
  -e SEMEMBED_MODEL=BAAI/bge-base-en-v1.5 \
  semstreams-semembed:latest
```

## Architecture

```
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
EMBEDDING_HTTP_MODEL=BAAI/bge-small-en-v1.5
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
task run:base           # bge-base-en-v1.5 (higher quality)
task run:minilm         # all-MiniLM-L6-v2 (faster)

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
  -d '{"input": ["test text"], "model": "BAAI/bge-small-en-v1.5"}'
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

## Migration from Old Go+ONNX Implementation

The new Rust service replaces the previous Go implementation that used:
- `knights-analytics/hugot` for ONNX inference
- Manual model downloading
- Custom HTTP API
- TEI for containerized deployments (linux/amd64 only)

**Benefits of new implementation**:
1. **Simpler**: fastembed-rs handles all complexity
2. **Faster**: Native Rust performance
3. **Smaller**: Lower memory footprint
4. **Multi-arch**: Works on ARM64 and AMD64 natively
5. **Standard**: OpenAI-compatible API

## Performance

Typical performance on various platforms:

| Platform | Model | Latency (single) | Throughput (batch-32) |
|----------|-------|------------------|----------------------|
| Apple M1 | bge-small-en-v1.5 | ~5ms | ~150ms |
| AMD64 4-core | bge-small-en-v1.5 | ~10ms | ~300ms |
| ARM64 2-core | bge-small-en-v1.5 | ~20ms | ~600ms |

*Benchmarks are approximate and depend on text length and hardware*

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
- Security scanning (Trivy)
- Dockerfile linting (Hadolint)

**Workflow**: `.github/workflows/ci.yml`

**Run CI locally**:
```bash
task ci:test
```

## License

Same as parent SemStreams project.

## Related

- [QUICKSTART.md](./QUICKSTART.md) - 5-minute getting started guide
- [Taskfile.yml](./Taskfile.yml) - Development automation tasks
- [fastembed-rs](https://github.com/Anush008/fastembed-rs) - Rust embedding library
- [SemStreams](../semstreams/) - Core framework
- [Graph Processor](../semstreams/processor/graph/) - Primary consumer
- [Taskfile Documentation](https://taskfile.dev) - Task runner docs
