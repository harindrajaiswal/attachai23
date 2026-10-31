# ORBIT AI — Multi-Tenant Grounded Customer Support SaaS Platform

**Orbit AI** is a production-oriented, multi-tenant customer-support AI SaaS platform built with Python (`FastAPI`, `Uvicorn`, `Pydantic v2`), `Supabase` (`PostgreSQL`, `pgvector`, `Supabase Storage`) with a deterministic local `SQLite` adapter, a server-rendered `Jinja2` Admin Control Center, and an embeddable vanilla JavaScript customer chat widget (`widget.js`).

Every tenant receives isolated knowledge ingestion, versioned 128-dimensional vector retrieval, deterministic post-generation grounding validation, conversation memory with automatic summarization, idempotent lead capture with resilient notification outbox retry, and configurable widget branding.

---

## Key Capabilities

1. **Strict Multi-Tenant Isolation**:
   - Every database table, vector query, storage object path (`{client_id}/{file_id}_{safe_filename}`), conversation, lead, and analytics query is strictly scoped by `client_id`.
   - Domain allowlist validation (`Origin` / `Referer` header normalization) and SHA-256 hashed tenant API keys (`X-API-Key`).
2. **Evidence-Grounded RAG & Hallucination Prevention**:
   - Multi-format document ingestion (`.txt`, `.md`, `.csv`, `.pdf`, `.docx`) with sentence-aware overlapping chunking.
   - Safe versioned re-indexing: new chunks are staged under a candidate `index_version` and atomically promoted only after complete embedding verification.
   - Hybrid vector + lexical retrieval with diversity deduplication and support confidence classification (`supported`, `weak`, `unsupported`).
   - Deterministic post-generation grounding validator (`app/services/grounding.py`) that rejects invented emails, phone numbers, URLs, numeric/pricing claims, or unauthorized policy promises before any token reaches the customer.
3. **Validated Streaming & Synchronous Chat**:
   - `/v1/chat` and `/v1/chat/stream` (SSE) both enforce grounding validation before emitting final customer-facing answers.
   - Automatic conversation summarization triggered every `SUMMARY_TRIGGER_MESSAGES` messages.
4. **Idempotent Lead Capture & Outbox Notifications**:
   - Duplicate submissions within the idempotency window return the existing lead without duplicating records or notifications.
   - Failed webhook/email notifications are persisted in the `notifications` outbox table with exponential backoff and manual/scheduled retry.
5. **Embeddable Customer Widget (`widget.js`)**:
   - Drop-in `<script>` tag for any tenant website with dynamic theme synchronization, grounded source attribution badges, quick-reply prompts, optional browser voice I/O, and inline lead capture.

---

## Repository Structure

```text
orbit-ai/
├── .env.example
├── .gitignore
├── .python-version
├── Dockerfile
├── README.md
├── pytest.ini
├── render.yaml
├── requirements.txt
├── supabase/
│   └── migrations/
│       └── 0001_initial.sql
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── lifespan.py
│   ├── core/
│   │   ├── config.py
│   │   ├── logging.py
│   │   ├── security.py
│   │   ├── exceptions.py
│   │   ├── rate_limiter.py
│   │   └── dependencies.py
│   ├── domain/
│   │   ├── enums.py
│   │   ├── models.py
│   │   └── types.py
│   ├── middleware/
│   │   ├── authentication.py
│   │   ├── cors.py
│   │   └── request_context.py
│   ├── providers/
│   │   ├── database.py
│   │   ├── storage.py
│   │   ├── vector.py
│   │   ├── embeddings.py
│   │   ├── llm.py
│   │   ├── notifications.py
│   │   └── tts.py
│   ├── repositories/
│   │   ├── base.py
│   │   ├── sqlite.py
│   │   └── supabase.py
│   ├── services/
│   │   ├── clients.py
│   │   ├── prompts.py
│   │   ├── parser.py
│   │   ├── knowledge.py
│   │   ├── advanced_rag.py
│   │   ├── grounding.py
│   │   ├── chat.py
│   │   ├── conversations.py
│   │   ├── leads.py
│   │   ├── notifications.py
│   │   └── analytics.py
│   ├── api/
│   │   ├── public.py
│   │   ├── admin.py
│   │   ├── system.py
│   │   └── schemas.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── login.html
│   │   ├── dashboard.html
│   │   ├── clients.html
│   │   ├── knowledge.html
│   │   ├── conversations.html
│   │   ├── leads.html
│   │   ├── analytics.html
│   │   ├── settings.html
│   │   └── partials/
│   │       └── header.html
│   └── utils/
│       ├── chunking.py
│       ├── csv_parser.py
│       ├── pii.py
│       └── validators.py
├── static/
│   ├── css/
│   │   ├── dashboard.css
│   │   └── widget.css
│   └── js/
│       ├── dashboard.js
│       └── widget.js
├── public/
│   └── demo.html
├── tests/
│   ├── conftest.py
│   ├── test_authentication.py
│   ├── test_tenants.py
│   ├── test_knowledge.py
│   ├── test_rag.py
│   ├── test_grounding.py
│   ├── test_chat.py
│   ├── test_conversations.py
│   ├── test_leads.py
│   ├── test_storage.py
│   ├── test_supabase.py
│   ├── test_notifications.py
│   ├── test_widget.py
│   ├── test_security.py
│   └── test_end_to_end.py
└── docs/
    ├── ARCHITECTURE.md
    ├── API.md
    ├── DATABASE.md
    ├── ENV_VARIABLES.md
    ├── SUPABASE_SETUP.md
    ├── RENDER_DEPLOYMENT.md
    └── TROUBLESHOOTING.md
```

---

## Quick Start (Local Development)

1. **Copy Environment Variables**:
   ```bash
   cp .env.example .env
   ```
   By default, `.env.example` configures `DATABASE_PROVIDER=sqlite`, `STORAGE_PROVIDER=local`, `VECTOR_PROVIDER=local`, and `LLM_PROVIDER_TYPE=extractive` so the platform runs locally with zero external cloud dependencies.

2. **Install Python Dependencies** (when running in a standard virtualenv):
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the Server**:
   ```bash
   # With Uvicorn:
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

   # Or using the built-in zero-dependency runner:
   python3 -m app.main --host 0.0.0.0 --port 8000
   ```

4. **Access the Platform**:
   - **Admin Dashboard**: `http://localhost:8000/admin` (Default local Admin Key: `orbit-dev-admin-key-2026`)
   - **Live Customer Widget Demo**: `http://localhost:8000/demo.html`
   - **Health & Readiness**: `http://localhost:8000/health` and `http://localhost:8000/ready`

5. **Run Automated Tests**:
   ```bash
   python3 -m unittest discover -s tests -v
   ```

---

## Embedding the Customer Widget

Add the following snippet before the closing `</body>` tag of any tenant website:

```html
<script
  src="https://your-orbit-domain.com/static/js/widget.js"
  data-client-id="acme-cloud"
  data-api-key="orb_live_your_tenant_api_key"
  data-api-base="https://your-orbit-domain.com"
  defer>
</script>
```

---

## Documentation Index

- [Architecture & Data Flows (`docs/ARCHITECTURE.md`)](docs/ARCHITECTURE.md)
- [REST API Reference (`docs/API.md`)](docs/API.md)
- [Database Schema & pgvector RPC (`docs/DATABASE.md`)](docs/DATABASE.md)
- [Environment Variables Reference (`docs/ENV_VARIABLES.md`)](docs/ENV_VARIABLES.md)
- [Supabase Production Setup (`docs/SUPABASE_SETUP.md`)](docs/SUPABASE_SETUP.md)
- [Render Production Deployment (`docs/RENDER_DEPLOYMENT.md`)](docs/RENDER_DEPLOYMENT.md)
- [Troubleshooting & Runbook (`docs/TROUBLESHOOTING.md`)](docs/TROUBLESHOOTING.md)
