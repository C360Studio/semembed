# Multi-stage build for semembed service
# Supports linux/amd64 and linux/arm64
# Uses fastembed-rs which handles model download and ONNX complexity

# Stage 1: Cargo chef planner
FROM rust:1.85-slim AS chef
# Install build dependencies for OpenSSL and C++ (for ONNX Runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*
RUN cargo install cargo-chef
WORKDIR /app

# Stage 2: Prepare dependencies
FROM chef AS planner
COPY Cargo.toml ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Build dependencies (cached layer)
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Stage 4: Build application
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release --bin semembed

# Stage 5: Runtime image
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash semembed

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/semembed /usr/local/bin/semembed

# Set ownership
RUN chown -R semembed:semembed /app

# Switch to non-root user
USER semembed

# Environment variables with defaults
ENV SEMEMBED_MODEL=BAAI/bge-small-en-v1.5
ENV SEMEMBED_PORT=8081
ENV RUST_LOG=info

# Expose service port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the service
# Note: fastembed-rs will download the model on first startup to ~/.cache/fastembed
ENTRYPOINT ["/usr/local/bin/semembed"]
