# SemEmbed Quick Start

**No Rust installation needed - everything runs in Docker!**

## Prerequisites

- Docker
- [Task](https://taskfile.dev/#/installation) (optional but recommended)

```bash
# Install Task
brew install go-task                  # macOS
snap install task --classic           # Ubuntu/Linux
go install github.com/go-task/task/v3/cmd/task@latest  # Go
```

## 5-Minute Quick Start

```bash
cd semembed

# 1. Build and run service
task dev

# 2. Test embeddings (OpenAI-compatible API)
curl -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Semantic search is powerful",
    "model": "BAAI/bge-small-en-v1.5"
  }'

# 3. View logs
task logs

# 4. Clean up
task clean
```

## Common Tasks

```bash
# Build
task build              # Build Docker image
task build:no-cache     # Rebuild from scratch

# Run
task run                # Run in background
task run:fg             # Run in foreground (see logs)
task run:base           # Run with bge-base model (higher quality)
task run:minilm         # Run with MiniLM model (faster)
task stop               # Stop service

# Test
task test:health        # Health check
task test:embed         # Single embedding test
task test:batch         # Batch embeddings test
task test:all           # All tests

# Monitor
task logs               # Follow logs
task logs:tail          # Last 50 lines
task metrics            # View Prometheus metrics

# Development
task restart            # Restart service
task shell              # Open container shell
task dev:rebuild        # Clean + rebuild + run

# Cleanup
task clean              # Remove container
task clean:all          # Remove everything
```

## Docker Compose Workflow

```bash
# Start
docker compose up -d

# Logs
docker compose logs -f

# Stop
docker compose down
```

Or use Task shortcuts:
```bash
task compose:up
task compose:logs
task compose:down
```

## API Examples

### Single Embedding

```bash
curl -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "This is a test sentence",
    "model": "BAAI/bge-small-en-v1.5"
  }'
```

### Batch Embeddings

```bash
curl -X POST http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      "First document",
      "Second document",
      "Third document"
    ],
    "model": "BAAI/bge-small-en-v1.5"
  }'
```

### Health Check

```bash
curl http://localhost:8081/health
```

## Available Models

```bash
# Small model (default) - 384 dims, fast
task run

# Base model - 768 dims, higher quality
task run:base

# MiniLM model - 384 dims, very fast
task run:minilm
```

## Troubleshooting

### Service won't start
```bash
# Check logs
task logs:tail

# Rebuild from scratch
task clean:all
task build:no-cache
task run
```

### Port already in use
```bash
# Find what's using port 8081
lsof -i :8081

# Kill the process or change port in docker-compose.yml
```

### Model download slow/fails
```bash
# First run downloads model - be patient
# Check progress in logs
task logs

# If download fails, clean cache and retry
task clean:cache
task run
```

## Integration with SemStreams

The service is already configured in `../semstreams/docker-compose.services.yml`:

```bash
# Start from semstreams directory
cd ../semstreams
docker compose -f docker-compose.services.yml --profile embedding up -d
```

Or use the Task shortcut:
```bash
cd semembed
task compose:up
```

## Getting Help

```bash
# List all tasks
task --list

# Show detailed help
task help
```

## Quick Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/models` | GET | List loaded models |
| `/v1/embeddings` | POST | Generate embeddings (OpenAI-compatible) |
| `/metrics` | GET | Prometheus metrics |

**Service URL**: `http://localhost:8081`

**Default Model**: `BAAI/bge-small-en-v1.5` (384 dimensions)

**OpenAI Compatibility**: Drop-in replacement for OpenAI embeddings API

**Expected Latency**: 5-20ms per embedding (CPU)

**Memory Usage**: ~500MB with small model

## Next Steps

- Read [README.md](./README.md) for full documentation
- Check [Taskfile.yml](./Taskfile.yml) for all available tasks
- Review [.github/workflows/ci.yml](.github/workflows/ci.yml) for CI/CD setup
- Integrate with SemStreams via Go client (see README)
