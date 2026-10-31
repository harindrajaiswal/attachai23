# Orbit AI — REST API Reference

All responses include an `X-Request-ID` header. Error responses follow a unified JSON structure:

```json
{
  "error": {
    "code": "AUTHENTICATION_FAILED",
    "message": "Invalid or missing tenant API key.",
    "request_id": "req_91a8d2c4f01b",
    "details": {}
  }
}
```

---

## 1. System Endpoints

| Method | Path | Auth | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | Public | Liveness check returning service version, environment, and uptime. |
| `GET` | `/ready` | Public | Readiness check verifying database, storage, vector, and embedding dimensions. |
| `GET` | `/metrics` | Admin (`X-Admin-Key` or session) | Operational metrics (requests, latencies, tokens, leads, active tenants). |

---

## 2. Public Widget API (`/v1/*`)

All `/v1/*` endpoints require tenant authentication via the `X-API-Key` header (or `Authorization: Bearer <tenant_api_key>`) and validate the request `Origin` / `Referer` against the tenant's configured domain allowlist.

### `GET /v1/config?client_id={client_id}`
Returns sanitized public widget branding and feature flags for the tenant. Never exposes internal prompts, webhooks, or hashes.

**Response `200 OK`**:
```json
{
  "client_id": "acme-cloud",
  "name": "Acme Cloud Support",
  "widget_config": {
    "title": "Acme Cloud Concierge",
    "subtitle": "Grounded answers from verified docs",
    "primary_color": "#0F766E",
    "accent_color": "#0284C7",
    "position": "bottom-right",
    "welcome_message": "Hi! Ask me anything about Acme Cloud plans, SLA, or security.",
    "placeholder_text": "Ask a question about Acme Cloud...",
    "quick_replies": ["What are your pricing tiers?", "What is your uptime SLA?"],
    "enable_voice_input": true,
    "enable_tts": false,
    "show_sources": true,
    "lead_capture_enabled": true
  }
}
```

### `POST /v1/chat`
Synchronous grounded chat completion.

**Request Body**:
```json
{
  "client_id": "acme-cloud",
  "message": "What is the Pro plan monthly price and uptime SLA?",
  "session_id": "sess_visitor_101",
  "conversation_id": null
}
```

**Response `200 OK`**:
```json
{
  "conversation_id": "conv_7e9a3d10c2b4",
  "message_id": "msg_4b81f02a9c11",
  "answer": "The Pro plan costs $49 per month and includes a 99.9% uptime SLA.",
  "support_level": "supported",
  "confidence": 0.91,
  "grounded": true,
  "fallback_used": false,
  "suggest_lead_capture": false,
  "citations": [
    {
      "file_id": "file_1290ab",
      "filename": "pricing-and-sla.md",
      "chunk_index": 0,
      "score": 0.91,
      "snippet": "Pro Plan: $49 per month. Includes a 99.9% uptime SLA and priority support."
    }
  ]
}
```

### `POST /v1/chat/stream`
Server-Sent Events (`text/event-stream`) endpoint that validates grounding before streaming tokens to the client.

**SSE Events**:
- `event: metadata` — `{ "conversation_id": "...", "support_level": "supported", "confidence": 0.91, "citations": [...] }`
- `event: token` — `{ "delta": "The Pro plan " }`
- `event: done` — `{ "message_id": "...", "answer": "...", "grounded": true, "fallback_used": false, "suggest_lead_capture": false }`
- `event: error` — `{ "code": "...", "message": "..." }`

### `POST /v1/leads`
Idempotent customer lead submission. Requires at least a valid `email` or `phone`.

**Request Body**:
```json
{
  "client_id": "acme-cloud",
  "name": "Jordan Rivera",
  "email": "jordan@example.com",
  "phone": "+1-415-555-0192",
  "company": "Rivera Logistics",
  "message": "Interested in Enterprise SSO for 120 seats.",
  "conversation_id": "conv_7e9a3d10c2b4",
  "session_id": "sess_visitor_101",
  "idempotency_key": "idem_optional_client_token"
}
```

---

## 3. Admin API (`/api/admin/*`)

Requires `X-Admin-Key: <ADMIN_API_KEY>` header OR an authenticated `/admin` session cookie plus `X-CSRF-Token` on state-changing (`POST`, `PUT`, `PATCH`, `DELETE`) requests.

### Overview & Analytics
- `GET /api/admin/overview` — Global KPIs, active tenants, knowledge readiness, lead conversion, and recent audit activity.
- `GET /api/admin/analytics?client_id={optional}&days=30` — Time-series usage, token consumption, support-level breakdown, and fallback rates.

### Tenants (`clients`)
- `GET /api/admin/clients` — List all tenants with usage and knowledge counts.
- `POST /api/admin/clients` — Create a new tenant and return its initial `raw_api_key` (shown once).
- `GET /api/admin/clients/{client_id}` — Detailed tenant configuration, domains, active prompt, and settings.
- `PATCH /api/admin/clients/{client_id}` — Update tenant name, status, widget branding, fallback message, domains, or settings.
- `POST /api/admin/clients/{client_id}/duplicate` — Clone a tenant's configuration, prompt, and settings into a new tenant slug.
- `POST /api/admin/clients/{client_id}/rotate-key` — Generate a new API key and optionally revoke prior keys.
- `DELETE /api/admin/clients/{client_id}` — Delete tenant and cascade cleanup across storage, vectors, conversations, and leads.

### Prompt Versioning
- `GET /api/admin/clients/{client_id}/prompts` — List prompt history for a tenant.
- `POST /api/admin/clients/{client_id}/prompts` — Publish and activate a new system prompt version.
- `POST /api/admin/clients/{client_id}/prompts/{version}/activate` — Roll back to a previous prompt version.

### Knowledge Base & RAG Diagnostics
- `GET /api/admin/clients/{client_id}/knowledge` — List knowledge files and indexing status.
- `POST /api/admin/clients/{client_id}/knowledge/upload` — Upload a document (`multipart/form-data` or JSON base64/text payload).
- `POST /api/admin/clients/{client_id}/knowledge/{file_id}/reindex` — Perform safe versioned re-indexing of a file.
- `DELETE /api/admin/clients/{client_id}/knowledge/{file_id}` — Delete file, storage object, and all associated vector chunks.
- `POST /api/admin/clients/{client_id}/rag/test` — Run a diagnostic RAG & grounding test query from the Admin UI.

### Conversations & Messages
- `GET /api/admin/conversations?client_id={optional}&search={optional}` — Paginated list of conversations.
- `GET /api/admin/conversations/{conversation_id}` — Full message transcript, citations, grounding status, and conversation summary.

### Leads & Workflow CRM
- `GET /api/admin/leads?client_id={optional}&status={optional}` — List captured leads with workflow status (`new`, `contacted`, `qualified`, `closed`, `archived`).
- `PATCH /api/admin/leads/{lead_id}/workflow` — Update lead status, assignee, or internal notes.
- `GET /api/admin/leads/export.csv?client_id={optional}` — Export leads as RFC 4180 CSV.

### Outbox Notifications & Settings
- `GET /api/admin/notifications?client_id={optional}&status={optional}` — Inspect notification outbox records.
- `POST /api/admin/notifications/{notification_id}/retry` — Immediately retry a failed/pending notification.
- `POST /api/admin/notifications/process-queue` — Process all due outbox notifications.
- `GET /api/admin/audit-logs?limit=100` — Inspect immutable administrative audit trail.
