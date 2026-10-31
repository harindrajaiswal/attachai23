# Orbit AI — System Architecture & Data Flows

Orbit AI is architected with clean layered boundaries separating HTTP transport (`app/api/`), authentication & request middleware (`app/middleware/`), business orchestration services (`app/services/`), pluggable infrastructure providers (`app/providers/`), and persistence repositories (`app/repositories/`).

---

## 1. End-to-End Customer Chat & Grounded RAG Flow

When a customer sends a message from `static/js/widget.js` to `/v1/chat` or `/v1/chat/stream`, the request traverses tenant authentication, domain allowlist validation, rate limiting, hybrid retrieval, LLM generation, and deterministic post-generation grounding validation before any answer reaches the browser.

```mermaid
sequenceDiagram
    autonumber
    participant Widget as static/js/widget.js
    participant PublicAPI as app/api/public.py
    participant AuthMW as app/middleware/authentication.py
    participant RateLim as app/core/rate_limiter.py
    participant ChatSvc as app/services/chat.py
    participant RAGSvc as app/services/advanced_rag.py
    participant EmbProv as app/providers/embeddings.py
    participant VecProv as app/providers/vector.py
    participant LLMProv as app/providers/llm.py
    participant GroundSvc as app/services/grounding.py
    participant ConvSvc as app/services/conversations.py
    participant Repo as app/repositories/sqlite.py | supabase.py

    Widget->>PublicAPI: POST /v1/chat or /v1/chat/stream (X-API-Key, Origin)
    PublicAPI->>AuthMW: require_tenant_auth(client_id, api_key, origin)
    AuthMW->>Repo: verify_api_key(client_id, sha256(key)) & list_domains(client_id)
    AuthMW->>RateLim: check_rate_limit(tenant + IP)
    PublicAPI->>ChatSvc: handle_chat(client_id, message, session_id, conversation_id)
    ChatSvc->>ConvSvc: get_or_create_conversation() & append_user_message()
    ChatSvc->>RAGSvc: retrieve_evidence(client_id, query, top_k)
    RAGSvc->>EmbProv: embed_query(normalized_query) -> 128-D vector
    RAGSvc->>VecProv: search(client_id, query_embedding, top_k)
    VecProv->>Repo: search_vectors(client_id, active_index_version)
    RAGSvc-->>ChatSvc: RetrievalResult(support_level, confidence, evidence_chunks)
    alt support_level == UNSUPPORTED
        ChatSvc-->>PublicAPI: Safe Tenant Fallback + Lead Capture Suggestion
    else support_level in (SUPPORTED, WEAK)
        ChatSvc->>LLMProv: generate_grounded_response(system_prompt, evidence, query)
        ChatSvc->>GroundSvc: validate_grounded_response(candidate_answer, evidence_chunks)
        alt Grounding Validation Passed
            ChatSvc->>ConvSvc: append_assistant_message(grounded=True) & maybe_summarize()
            ChatSvc-->>PublicAPI: Validated Grounded Answer + Citations
        else Grounding Violation Detected (invented email/phone/URL/number/guarantee)
            ChatSvc->>GroundSvc: extract_deterministic_fallback(evidence_chunks)
            ChatSvc->>ConvSvc: append_assistant_message(repaired_or_fallback)
            ChatSvc-->>PublicAPI: Safe Extractive Answer or Tenant Fallback
        end
    end
    PublicAPI-->>Widget: JSON Response or Validated SSE Stream
```

---

## 2. Admin Control Center & Configuration Flow

Administrators authenticate via `X-Admin-Key` or an `HttpOnly` HMAC-signed session cookie at `/admin/login`. Every state-changing browser action validates a CSRF token (`X-CSRF-Token`) and writes an immutable entry to `audit_logs`.

```mermaid
flowchart TD
    AdminBrowser["Admin Browser (/admin + static/js/dashboard.js)"]
    AdminAPI["app/api/admin.py"]
    AuthMW["app/middleware/authentication.py"]
    ClientSvc["app/services/clients.py"]
    PromptSvc["app/services/prompts.py"]
    AnalyticsSvc["app/services/analytics.py"]
    Repo["app/repositories/sqlite.py | supabase.py"]

    AdminBrowser -->|POST /admin/login or X-Admin-Key + X-CSRF-Token| AdminAPI
    AdminAPI --> AuthMW
    AuthMW -->|Constant-Time hmac.compare_digest| AdminAPI
    AdminAPI -->|Create / Update / Duplicate / Rotate Key| ClientSvc
    AdminAPI -->|Create / Rollback Prompt Version| PromptSvc
    AdminAPI -->|Dashboard Overview & Usage Rollups| AnalyticsSvc
    ClientSvc -->|Persist + Audit Log| Repo
    PromptSvc -->|Versioned Prompt + Audit Log| Repo
    AnalyticsSvc -->|Scoped Aggregations| Repo
```

---

## 3. Safe Versioned Knowledge Upload & Re-Indexing Flow

Uploading or re-indexing a knowledge document never leaves a tenant with an empty or half-written vector index. Chunks are staged under a candidate `index_version` and promoted only after all embeddings are verified.

```mermaid
sequenceDiagram
    autonumber
    participant AdminUI as static/js/dashboard.js
    participant AdminAPI as app/api/admin.py
    participant KnowSvc as app/services/knowledge.py
    participant Parser as app/services/parser.py
    participant Storage as app/providers/storage.py
    participant Chunker as app/utils/chunking.py
    participant EmbProv as app/providers/embeddings.py
    participant VecProv as app/providers/vector.py
    participant Repo as app/repositories/sqlite.py | supabase.py

    AdminUI->>AdminAPI: POST /api/admin/clients/{client_id}/knowledge/upload
    AdminAPI->>KnowSvc: ingest_document(client_id, filename, raw_bytes)
    KnowSvc->>Repo: compute sha256 checksum & create knowledge_files(status=PROCESSING)
    KnowSvc->>Storage: save_file("{client_id}/{file_id}_{safe_filename}", raw_bytes)
    KnowSvc->>Parser: parse_document(filename, raw_bytes) (.txt/.md/.csv/.pdf/.docx)
    KnowSvc->>Chunker: chunk_text(parsed_text, chunk_size, overlap)
    KnowSvc->>EmbProv: embed_texts(chunks) -> list[list[float]] (128 dimensions)
    KnowSvc->>VecProv: upsert_chunks(client_id, file_id, candidate_index_version, chunks)
    alt Embedding & Vector Storage Succeeded
        KnowSvc->>VecProv: delete_file_vectors(client_id, file_id, exclude_version=candidate_version)
        KnowSvc->>Repo: update knowledge_files(status=READY, active_index_version=candidate_version)
        KnowSvc-->>AdminAPI: KnowledgeFile(status=READY, chunk_count)
    else Parsing or Embedding Failed
        KnowSvc->>VecProv: delete_file_vectors(client_id, file_id, only_version=candidate_version)
        KnowSvc->>Repo: restore prior active_index_version or mark FAILED with error_message
        KnowSvc-->>AdminAPI: Structured Error (previous active index preserved)
    end
```

---

## 4. Idempotent Lead Capture & Outbox Notification Flow

Lead capture is decoupled from external notification delivery so a transient webhook or email outage never loses a customer lead or causes duplicate rows.

```mermaid
sequenceDiagram
    autonumber
    participant Widget as static/js/widget.js
    participant PublicAPI as app/api/public.py
    participant LeadSvc as app/services/leads.py
    participant NotifSvc as app/services/notifications.py
    participant NotifProv as app/providers/notifications.py
    participant Repo as app/repositories/sqlite.py | supabase.py

    Widget->>PublicAPI: POST /v1/leads (client_id, email/phone, idempotency_key)
    PublicAPI->>LeadSvc: capture_lead(client_id, payload)
    LeadSvc->>Repo: find_lead_by_idempotency_key(client_id, computed_key)
    alt Existing Lead Found Within Window
        Repo-->>LeadSvc: Existing Lead Record
        LeadSvc-->>PublicAPI: Return Existing Lead (is_duplicate=True)
    else New Lead
        LeadSvc->>Repo: create_lead(lead) & create_lead_workflow(status=NEW)
        LeadSvc->>NotifSvc: enqueue_and_dispatch_lead_notifications(client, lead)
        NotifSvc->>Repo: create_notification(status=PENDING)
        NotifSvc->>NotifProv: dispatch(webhook / email)
        alt Delivery Succeeded
            NotifSvc->>Repo: update_notification(status=SENT, delivered_at=now)
        else Delivery Failed
            NotifSvc->>Repo: update_notification(status=RETRYING, next_retry_at=backoff)
        end
        LeadSvc-->>PublicAPI: Created Lead Record (HTTP 201)
    end
```
