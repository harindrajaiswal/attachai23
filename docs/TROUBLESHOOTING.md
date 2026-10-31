# Orbit AI — Troubleshooting & Operational Runbook

This guide covers common operational scenarios, diagnostics, and resolutions.

---

## 1. Startup Configuration Errors (`ConfigurationError`)

### Symptom:
Application refuses to start in `production` mode with:
- `ADMIN_API_KEY must be changed from the default development key in production.`
- `ALLOWED_ORIGINS cannot be '*' in production.`
- `EMBEDDING_DIMENSIONS (X) must match SUPABASE_VECTOR_DIMENSIONS (128).`

### Resolution:
- Set a strong random `ADMIN_API_KEY` (at least 24 characters).
- Set `ALLOWED_ORIGINS` to explicit comma-separated origins (e.g., `https://app.example.com`).
- Ensure both `EMBEDDING_DIMENSIONS=128` and `SUPABASE_VECTOR_DIMENSIONS=128`.

---

## 2. Widget Requests Blocked with `403 DOMAIN_NOT_ALLOWED`

### Symptom:
The customer chat widget loads on a tenant website, but `/v1/config` or `/v1/chat` returns HTTP `403`.

### Resolution:
1. Open `/admin/clients` and inspect the tenant's **Allowed Domains**.
2. Add the exact hostname where the widget is hosted (e.g., `shop.example.com` or `*.example.com`).
3. Orbit AI normalizes `Origin` and `Referer` headers to lowercase hostnames without ports or trailing slashes.

---

## 3. Chat Answers Return the Tenant Fallback Message

### Symptom:
`/v1/chat` returns the tenant's fallback message (`fallback_used: true`) instead of a specific answer.

### Resolution:
1. Open `/admin/knowledge`, select the tenant, and verify that uploaded files have status **`ready`** and `chunk_count > 0`.
2. Use the **RAG & Grounding Diagnostic Tester** on `/admin/knowledge` to inspect the retrieved chunks, similarity scores, and grounding validator report:
   - If `support_level` is `unsupported`, upload documentation covering that topic or lower `similarity_threshold` in `/admin/settings`.
   - If the grounding validator rejected the LLM candidate answer, inspect `violations` (e.g., an upstream LLM tried to invent a phone number, URL, or numeric price not present in the retrieved chunks).

---

## 4. Document Upload Fails (`400` or `422`)

### Symptom:
Knowledge file upload fails in `/admin/knowledge`.

### Resolution:
- **Unsupported extension**: Only `.txt`, `.md`, `.csv`, `.pdf`, and `.docx` are accepted.
- **File size**: Maximum upload size is `MAX_UPLOAD_SIZE_BYTES` (default 5 MB).
- **Empty text**: Scanned image-only PDFs without extractable text are rejected so empty chunks never pollute the vector index. Convert image PDFs with OCR to `.txt` or `.md` before uploading.

---

## 5. Lead Notifications Stuck in `retrying` or `failed`

### Symptom:
A customer lead appears in `/admin/leads`, but the webhook notification shows `retrying` or `failed` in `/admin/settings`.

### Resolution:
- Because Orbit AI persists leads before dispatching notifications, **no lead data is lost**.
- Verify the tenant's `webhook_url` in `/admin/settings` is reachable and returns HTTP `2xx`.
- Click **Retry** next to the notification in `/admin/settings` or call `POST /api/admin/notifications/process-queue` to flush due retries.
