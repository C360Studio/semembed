# Multi-stage build for semembed service
# Supports linux/amd64 and linux/arm64
# Uses fastembed-rs which handles model download and ONNX complexity

# Stage 1: Cargo chef planner
# Builder and runtime are both pinned to Debian Trixie. fastembed's
# prebuilt ONNX Runtime requires glibc 2.38+ and libstdc++13 symbols
# (__isoc23_strtol, __cxa_call_terminate), so an older base on either
# side breaks linking or execution.
FROM rust:1.95-slim-trixie AS chef
# Install build dependencies for OpenSSL and C++ (for ONNX Runtime)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Pin cargo-chef for reproducible dependency recipe builds
RUN cargo install cargo-chef --locked --version 0.1.78
WORKDIR /app

# Stage 2: Prepare dependencies
FROM chef AS planner
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo chef prepare --recipe-path recipe.json

# Stage 3: Build dependencies (cached layer)
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --locked --recipe-path recipe.json

# Stage 4: Build application
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release --locked --bin semembed

# Unit tests (CI builds this target; reuses the cached release dependencies)
FROM builder AS test
RUN cargo test --release --locked --bin semembed

# Stage 5: Runtime image
FROM debian:trixie-slim AS runtime

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

# Set ownership and prepare model cache directory
RUN mkdir -p /home/semembed/.cache/fastembed && \
    chown -R semembed:semembed /app /home/semembed/.cache

# Switch to non-root user
USER semembed

# Environment variables with defaults
ENV SEMEMBED_MODEL=Snowflake/snowflake-arctic-embed-s
ENV SEMEMBED_PORT=8081
ENV FASTEMBED_CACHE_DIR=/home/semembed/.cache/fastembed
ENV RUST_LOG=info

# Expose service port
EXPOSE 8081

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Run the service
# Note: fastembed-rs downloads the model on first startup to FASTEMBED_CACHE_DIR
ENTRYPOINT ["/usr/local/bin/semembed"]
