# Orbit AI — Database Schema & pgvector Reference

Orbit AI uses a canonical 14-table relational + vector schema defined in [`supabase/migrations/0001_initial.sql`](../supabase/migrations/0001_initial.sql) and mirrored in `app/repositories/sqlite.py` for zero-dependency local execution.

---

## 1. Required Extensions (Supabase / PostgreSQL)

- `pgcrypto` — UUID generation (`gen_random_uuid()`) and cryptographic helpers.
- `vector` (`pgvector`) — Fixed 128-dimensional embedding storage (`vector(128)`) and cosine similarity indexing (`vector_cosine_ops`).

---

## 2. The 14 Core Tables

| # | Table Name | Primary Key | Tenant Key | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `clients` | `id` (`TEXT`) | `id` | Tenant identity, status (`active`/`suspended`/`disabled`), widget theme JSON, fallback text, and active prompt pointer. |
| 2 | `api_keys` | `id` (`TEXT`) | `client_id` | SHA-256 hashed tenant API keys (`key_hash`), display prefix (`key_prefix`), active status, and last-used timestamp. |
| 3 | `domains` | `id` (`TEXT`) | `client_id` | Normalized allowed origins/hostnames per tenant (`domain`, `is_active`). |
| 4 | `prompt_versions` | `id` (`TEXT`) | `client_id` | Immutable versioned system prompts (`version`, `system_prompt`, `style_guidelines`, `is_active`). |
| 5 | `conversations` | `id` (`TEXT`) | `client_id` | Customer chat sessions (`session_id`, `status`, `summary`, `message_count`, `metadata`). |
| 6 | `messages` | `id` (`TEXT`) | `client_id` | Individual turns (`conversation_id`, `role`, `content`, `support_level`, `confidence`, `grounded`, `citations`, `tokens_used`). |
| 7 | `leads` | `id` (`TEXT`) | `client_id` | Captured customer leads (`name`, `email`, `phone`, `company`, `message`, `idempotency_key` unique per tenant). |
| 8 | `lead_workflow` | `id` (`TEXT`) | `client_id` | CRM workflow state per lead (`lead_id`, `status`, `assigned_to`, `notes`, `updated_by`). |
| 9 | `usage_logs` | `id` (`TEXT`) | `client_id` | Per-operation telemetry (`operation`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`, `support_level`, `fallback_used`). |
| 10 | `audit_logs` | `id` (`TEXT`) | `client_id` (nullable) | Immutable administrative audit log (`actor`, `action`, `resource_type`, `resource_id`, `details`, `ip_address`). |
| 11 | `knowledge_files` | `id` (`TEXT`) | `client_id` | Uploaded knowledge file metadata (`filename`, `storage_path`, `mime_type`, `file_size`, `checksum`, `status`, `chunk_count`, `active_index_version`). |
| 12 | `knowledge_vectors` | `id` (`TEXT`) | `client_id` | Indexed document chunks (`file_id`, `index_version`, `chunk_index`, `content`, `token_count`, `embedding vector(128)`, `metadata`). |
| 13 | `settings` | `id` (`TEXT`) | `client_id` | Per-tenant RAG thresholds, rate limits, notification webhooks, and email recipients. |
| 14 | `notifications` | `id` (`TEXT`) | `client_id` | Transactional notification outbox (`lead_id`, `channel`, `recipient`, `payload`, `status`, `attempt_count`, `next_retry_at`, `last_error`). |

---

## 3. Vector Dimension Contract (`vector(128)`)

To prevent dimension mismatch errors between the embedding provider and `pgvector`:
- `SUPABASE_VECTOR_DIMENSIONS` defaults to `128`.
- `knowledge_vectors.embedding` is declared as `vector(128)`.
- `app/providers/embeddings.py` validates that every generated embedding vector has `len(vec) == settings.supabase_vector_dimensions` (128).
- `app/core/config.py` validates at startup that `EMBEDDING_DIMENSIONS == SUPABASE_VECTOR_DIMENSIONS == 128`.

---

## 4. `match_knowledge_vectors` RPC Function

Defined in `supabase/migrations/0001_initial.sql`:

```sql
CREATE OR REPLACE FUNCTION public.match_knowledge_vectors(
    p_client_id TEXT,
    p_query_embedding vector(128),
    p_match_count INTEGER DEFAULT 5,
    p_min_similarity DOUBLE PRECISION DEFAULT 0.15
)
RETURNS TABLE (
    id TEXT,
    client_id TEXT,
    file_id TEXT,
    filename TEXT,
    index_version INTEGER,
    chunk_index INTEGER,
    content TEXT,
    token_count INTEGER,
    metadata JSONB,
    similarity DOUBLE PRECISION
)
```

The RPC joins `knowledge_vectors` with `knowledge_files` and strictly filters:
1. `kv.client_id = p_client_id` AND `kf.client_id = p_client_id`
2. `kf.status = 'ready'`
3. `kv.index_version = kf.active_index_version` (ensuring staged or failed candidate versions are never returned)
4. `(1 - (kv.embedding <=> p_query_embedding)) >= p_min_similarity`
