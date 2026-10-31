# Orbit AI — Render Production Deployment Guide

Orbit AI includes a production-ready [`render.yaml`](../render.yaml) Blueprint and [`Dockerfile`](../Dockerfile) for one-click or automated deployment to [Render](https://render.com).

---

## Option A: Deploy via Render Blueprint (`render.yaml`)

1. Push this repository to your GitHub or GitLab account.
2. In the [Render Dashboard](https://dashboard.render.com), click **New + -> Blueprint**.
3. Connect your repository. Render will automatically detect `render.yaml` and configure the `orbit-ai` Python web service with:
   - Build Command: `pip install --upgrade pip && pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT --workers 2 --proxy-headers`
   - Health Check Path: `/health`
4. Prompted secret variables (`sync: false` in `render.yaml`):
   - `ALLOWED_ORIGINS`: Set to your production domain(s), e.g., `https://support.yourdomain.com,https://yourdomain.com` (never `*` in production).
   - `SUPABASE_URL`: Your Supabase project URL.
   - `SUPABASE_ANON_KEY`: Your Supabase anon key.
   - `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role secret key.
   - `LLM_API_KEY`: (Optional) Your OpenAI-compatible API key if `LLM_PROVIDER_TYPE=openai_compatible`.
5. `ADMIN_API_KEY` and `SESSION_SECRET_KEY` are automatically generated as strong cryptographic values by Render (`generateValue: true`). You can view `ADMIN_API_KEY` in the **Environment** tab of your Render service to sign into `/admin`.

---

## Option B: Deploy via Docker (`Dockerfile`)

Orbit AI also ships with a hardened non-root `Dockerfile`:

```bash
docker build -t orbit-ai:latest .
docker run -p 8000:8000 --env-file .env orbit-ai:latest
```

---

## Post-Deployment Verification Checklist

1. **Health Check**:
   ```bash
   curl -i https://your-orbit-service.onrender.com/health
   ```
2. **Readiness Check**:
   ```bash
   curl -i https://your-orbit-service.onrender.com/ready
   ```
3. **Admin Login**:
   - Open `https://your-orbit-service.onrender.com/admin`
   - Enter your `ADMIN_API_KEY` from the Render Environment tab.
4. **Create First Production Tenant**:
   - Create a tenant in `/admin/clients`, configure allowed domains, upload a knowledge document in `/admin/knowledge`, and copy the embed snippet into your website.
