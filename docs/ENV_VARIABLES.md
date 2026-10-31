# Orbit AI — Environment Variables Reference

All configuration variables are loaded and validated by `app/core/config.py` at startup. Copy `.env.example` to `.env` for local development.

---

## Core Application & Security

| Variable | Required in Prod | Default (Local) | Description |
| :--- | :--- | :--- | :--- |
| `APP_NAME` | No | `Orbit AI` | Application display name. |
| `APP_ENV` | Yes | `development` | `development`, `staging`, `test`, or `production`. |
| `PORT` | No | `8000` | HTTP listen port. |
| `LOG_LEVEL` | No | `INFO` | Structured JSON logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `ADMIN_API_KEY` | **Yes** | `orbit-dev-admin-key-2026` | Master administrative key for `/admin` and `/api/admin/*`. Must be strong in production. |
| `SESSION_SECRET_KEY` | **Yes** | *(derived in dev)* | Secret key used to sign `HttpOnly` admin session cookies and CSRF tokens. |
| `ALLOWED_ORIGINS` | **Yes** | `*` | Comma-separated global CORS origins. In `production`, wildcard `*` is rejected at startup. |
| `PROTECT_METRICS` | No | `true` | Require admin authentication on `/metrics`. |

---

## Pluggable Infrastructure Providers

| Variable | Allowed Values | Default (Local) | Description |
| :--- | :--- | :--- | :--- |
| `DATABASE_PROVIDER` | `sqlite`, `supabase` | `sqlite` | Persistence backend for all 14 tables. |
| `STORAGE_PROVIDER` | `local`, `supabase` | `local` | Object storage backend for uploaded knowledge files. |
| `VECTOR_PROVIDER` | `local`, `supabase` | `local` | Vector similarity search provider. |
| `SQLITE_DB_PATH` | File path | `./data/orbit_local.db` | Local SQLite database file path when `DATABASE_PROVIDER=sqlite`. |
| `LOCAL_STORAGE_PATH` | Directory path | `./data/storage` | Local directory for knowledge files when `STORAGE_PROVIDER=local`. |

---

## Supabase Configuration (Required when `*_PROVIDER=supabase`)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `SUPABASE_URL` | `""` | Supabase project URL (`https://<project_ref>.supabase.co`). |
| `SUPABASE_ANON_KEY` | `""` | Supabase public anonymous key. |
| `SUPABASE_SERVICE_ROLE_KEY` | `""` | Server-only Supabase Service Role key (never exposed to clients). |
| `SUPABASE_STORAGE_BUCKET` | `knowledge` | Private Supabase Storage bucket for tenant documents. |
| `SUPABASE_VECTOR_RPC` | `match_knowledge_vectors` | PostgreSQL RPC function name for tenant-scoped cosine similarity search. |
| `SUPABASE_VECTOR_DIMENSIONS` | `128` | Vector embedding dimension matching `vector(128)` in SQL migration. |

---

## Embeddings, RAG, & LLM Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `EMBEDDING_PROVIDER_TYPE` | `deterministic` | `deterministic` (built-in 128-D semantic hash + n-gram embedder) or `openai_compatible`. |
| `EMBEDDING_DIMENSIONS` | `128` | Must match `SUPABASE_VECTOR_DIMENSIONS` (`128`). |
| `LLM_PROVIDER_TYPE` | `extractive` | `extractive` (deterministic grounded synthesizer) or `openai_compatible` (OpenAI / Groq / Together / OpenRouter / Gemini OpenAI-compatible endpoint). |
| `LLM_API_KEY` | `""` | API key when `LLM_PROVIDER_TYPE=openai_compatible`. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Base URL for OpenAI-compatible chat completions. |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier for grounded generation. |
| `LLM_TIMEOUT_SECONDS` | `20` | Per-request timeout for upstream LLM HTTP calls. |
| `LLM_MAX_RETRIES` | `2` | Automatic retry count on transient `429` / `5xx` upstream errors. |
| `RAG_TOP_K` | `5` | Maximum number of chunks retrieved per query. |
| `RAG_SIMILARITY_THRESHOLD` | `0.18` | Minimum hybrid score required to consider a chunk relevant. |
| `RAG_SUPPORTED_THRESHOLD` | `0.42` | Minimum top score for `supported` confidence level. |
| `CHUNK_SIZE_CHARS` | `700` | Target character length per document chunk. |
| `CHUNK_OVERLAP_CHARS` | `120` | Character overlap between adjacent chunks. |
| `MAX_UPLOAD_SIZE_BYTES` | `5242880` | Maximum file upload size (default 5 MB). |
| `SUMMARY_TRIGGER_MESSAGES` | `6` | Automatically refresh conversation summary every N messages. |

---

## Rate Limiting & Outbox Notifications

| Variable | Default | Description |
| :--- | :--- | :--- |
| `RATE_LIMIT_CHAT_PER_MINUTE` | `60` | Maximum `/v1/chat` requests per minute per tenant + IP. |
| `RATE_LIMIT_LEADS_PER_MINUTE` | `20` | Maximum `/v1/leads` submissions per minute per tenant + IP. |
| `LEAD_IDEMPOTENCY_WINDOW_HOURS` | `24` | Deduplication window for identical lead submissions. |
| `WEBHOOK_TIMEOUT_SECONDS` | `8` | HTTP timeout for outbound lead notification webhooks. |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | Optional | Optional SMTP relay configuration for email lead alerts. |
