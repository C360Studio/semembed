# CLAUDE.md — semembed

Project-specific instructions. Overrides the global `~/.claude/CLAUDE.md`
(which is written for Go/Svelte/TypeScript work and does not apply here).

## What this is

`semembed` is a small Rust HTTP service that exposes an OpenAI-compatible
`/v1/embeddings` endpoint backed by [`fastembed-rs`]. It runs an ONNX
embedding model (default `Snowflake/snowflake-arctic-embed-s`) in-process
and ships as a single static binary in a Debian slim container.

The service is intentionally minimal — one binary, one source file
(`src/main.rs`), no database, no auth layer. Treat it as a sidecar.

[`fastembed-rs`]: https://github.com/Anush008/fastembed-rs

## Stack

- **Language**: Rust 2021 edition, toolchain pinned to `1.95` (Dockerfile + CI)
- **HTTP**: `axum` 0.8 + `tower-http` 0.6 (CORS + tracing layers)
- **Async runtime**: `tokio` 1.x with `full` features
- **Embeddings**: `fastembed` 7.x (downloads model to `~/.cache/fastembed` on first run)
- **Metrics**: `prometheus` 0.14 exposed at `/metrics`
- **Logging**: `tracing` + `tracing-subscriber` with `RUST_LOG` env filter
- **Errors**: `anyhow::Result` everywhere; no custom error types

## Local workflow

You almost certainly do **not** have a local `cargo` toolchain. Verify
changes by building the Docker image — that is the project's canonical
build path and matches CI exactly:

```sh
docker build -t semembed:dev .
```

The build uses `cargo-chef` for dep caching across stages, so iterative
edits to `src/main.rs` only rebuild the final layer.

A `Taskfile.yml` exists for common operations — check it before
hand-rolling commands.

## Code conventions

- Keep everything in `src/main.rs` until it genuinely needs splitting.
  Premature module structure is the bigger risk than a long single file
  for a service this small.
- Use `anyhow::Result<T>` for fallible functions. Do not introduce
  `thiserror` or hand-rolled error enums unless an error needs to cross
  a public API boundary (it currently does not).
- Pass `&Arc<AppState>` via `axum::extract::State`. Do not stash global
  state in `static`/`OnceCell` — the existing `State` extractor pattern
  is the convention.
- The `TextEmbedding` model is wrapped in a `std::sync::Mutex` because
  `embed()` takes `&mut self`. This serializes inference. If you need
  concurrency, prefer a small pool over swapping in `tokio::sync::Mutex`
  (the work is CPU-bound, not await-bound).
- Log with `tracing::{info, warn, error}` macros, not `println!`.
- Metrics: register on the per-service `Registry` (not the global
  default registry) so `/metrics` returns only this service's metrics.

## API contract — do not break casually

The `/v1/embeddings` endpoint mimics OpenAI's embeddings API so callers
can point an OpenAI SDK at this service unchanged. That means:

- Request: `{ "input": string | string[], "model": string?, "encoding_format": "float" | "base64"?, "input_type": "query" | "document"? }`
- Response shape: `{ object: "list", data: [{object, embedding, index, truncated?: true}], model, usage: {prompt_tokens, total_tokens}, prefix_applied: bool }`
- Error shape: `{ error: { message, type } }`

`input_type` defaults to `"document"` when omitted or `null`, applying the
served model's query or document prefix from `ModelSpec` in `src/main.rs`
(without double-prefixing inputs that already carry a known prefix for that
model). `truncated` is omitted when `false` and emitted as `true` only on
entries cut to `max_tokens` (also counted in `semembed_inputs_truncated_total`).

`usage.prompt_tokens` and `semembed_tokens_processed_total` report real
model token counts from the active HuggingFace tokenizer (including special
tokens and any applied prefix, capped at `max_tokens` when truncated).

`encoding_format: "base64"` is accepted in the request schema but the
response is always float arrays. Either implement base64 properly or
reject the value — don't silently ignore it long-term.

## Models

The `SEMEMBED_MODEL` env var picks the model, and optional
`SEMEMBED_MAX_TOKENS` lowers the context window below the model's native
limit (clamped if set higher). Anything not matched in `src/main.rs` logs a
warning and falls back to the default (`SnowflakeArcticEmbedS`). When
adding a model, add its `ModelSpec` arm (including `native_max_tokens`,
`query_prefix`, and `document_prefix`) *and* update the README's
supported-models table.

**Currently mapped**:

- `Snowflake/snowflake-arctic-embed-s` → `SnowflakeArcticEmbedS` *(default — 384d, 512 ctx)*
- `Snowflake/snowflake-arctic-embed-xs` → `SnowflakeArcticEmbedXS` (384d, 512 ctx)
- `Snowflake/snowflake-arctic-embed-m` → `SnowflakeArcticEmbedM` (768d, 512 ctx)
- `Snowflake/snowflake-arctic-embed-m-long` → `SnowflakeArcticEmbedMLong` (768d, 8192 ctx)
- `nomic-ai/nomic-embed-text-v1.5` → `NomicEmbedTextV15` (768d MRL, 8192 ctx)
- `google/embeddinggemma-300m` (`onnx-community/embeddinggemma-300m-ONNX`) → `EmbeddingGemma300M` (768d MRL, 2048 ctx)
- `google/embeddinggemma-300m-q` → `EmbeddingGemma300MQ` (768d MRL, 2048 ctx)
- `google/embeddinggemma-300m-q4` → `EmbeddingGemma300MQ4` (768d MRL, 2048 ctx)
- `lightonai/modernbert-embed-large` → `ModernBertEmbedLarge` (1024d, 8192 ctx)
- `BAAI/bge-small-en-v1.5` → `BGESmallENV15` (384d, 512 ctx, prior default)
- `BAAI/bge-base-en-v1.5` → `BGEBaseENV15` (768d, 512 ctx)
- `sentence-transformers/all-MiniLM-L6-v2` → `AllMiniLML6V2` (384d, 256 ctx)

### Why the default is `snowflake-arctic-embed-s`

Same shape as the prior `bge-small-en-v1.5` default (33M params, 384-dim,
512-token context, Apache-2.0) and matches downstream 384d index
requirements, with a slightly higher MTEB Retrieval score
(NDCG@10 51.98 vs 51.68) and from the actively-maintained Snowflake
suite that scales up cleanly to `-m` (110M, 768-dim), `-m-long` (137M,
8K context), and `-l` (335M) without changing tokenizer family. The lead over bge-small is small
(~0.30 NDCG) — the better argument is that the Snowflake suite is the
direction the niche is moving and gives us a coherent upgrade path.

`fastembed-rs` v7 exposes 40+ text embedding variants total (see
`docs.rs/fastembed`); only the models above are reachable today because
`SEMEMBED_MODEL` strings are matched literally. Add new arms when a
caller needs them — don't auto-expose everything.

### Vector-space compatibility

Switching models produces incompatible embedding vectors — they are not
mathematically comparable across model families. semstreams (the
downstream consumer) is currently in beta/greenfield with no persisted
vectors, so default-model changes here are free. **If that changes,
adding a new default becomes a breaking change** requiring a re-embed
plan downstream.

## Dependency upgrades

- `Cargo.lock` is committed and enforced via `--locked` in `Dockerfile`.
- Compatible bumps (caret-allowed): edit `Cargo.toml`, regenerate
  `Cargo.lock` inside `rust:1.95-slim-trixie`, run a docker build, ship.
- Major bumps (axum, tower-http, prometheus, fastembed, thiserror-style
  crates): check the changelog for the specific APIs used in
  `src/main.rs` before bumping. The surface area is small — usually a
  10-minute audit.
- Rust toolchain: bump in **both** `Dockerfile` (`FROM rust:X.Y-slim-trixie`)
  and `.github/workflows/ci.yml` (`RUST_VERSION`) together.

## What to skip

- Don't add a database, auth, or rate-limiting layer here. Those belong
  upstream (gateway / sidecar consumers).
- Don't introduce `async-trait`, custom executors, or actor frameworks.
  The shape of this service does not need them.
- Don't add unit tests that mock `TextEmbedding` — the model *is* the
  behavior. Integration tests against a running container (as CI does)
  are the right level.
