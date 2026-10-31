# Orbit AI — Supabase Production Setup Guide

Follow these steps to configure Supabase (`PostgreSQL`, `pgvector`, and `Supabase Storage`) as the production backend for Orbit AI.

---

## Step 1: Create a Supabase Project

1. Sign in to [https://supabase.com/dashboard](https://supabase.com/dashboard) and create a new project.
2. Navigate to **Project Settings -> API** and copy:
   - **Project URL** (`SUPABASE_URL`)
   - **anon public key** (`SUPABASE_ANON_KEY`)
   - **service_role secret key** (`SUPABASE_SERVICE_ROLE_KEY`) — *Keep this secret on the backend server only.*

---

## Step 2: Run the Initial SQL Migration

1. Open the **SQL Editor** in your Supabase Dashboard (or use the Supabase CLI).
2. Copy the entire contents of [`supabase/migrations/0001_initial.sql`](../supabase/migrations/0001_initial.sql) and execute it.
3. This single migration idempotently provisions:
   - Extensions: `pgcrypto` and `vector` (`pgvector`)
   - All 14 required tables (`clients`, `api_keys`, `domains`, `prompt_versions`, `conversations`, `messages`, `leads`, `lead_workflow`, `usage_logs`, `audit_logs`, `knowledge_files`, `knowledge_vectors`, `settings`, `notifications`)
   - HNSW cosine index on `knowledge_vectors.embedding vector(128)`
   - Tenant-scoped RPC function `public.match_knowledge_vectors(p_client_id, p_query_embedding, p_match_count, p_min_similarity)`
   - Row Level Security (RLS) policies on all 14 tables
   - Private `knowledge` bucket in `storage.buckets`

---

## Step 3: Verify the Schema & Vector RPC

Run the following verification query in the Supabase SQL Editor:

```sql
-- Verify all 14 tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
ORDER BY table_name;

-- Verify the 128-D vector RPC exists
SELECT proname, proargnames
FROM pg_proc
WHERE proname = 'match_knowledge_vectors';

-- Verify the private 'knowledge' storage bucket exists
SELECT id, name, public
FROM storage.buckets
WHERE id = 'knowledge';
```

---

## Step 4: Configure Orbit AI Environment Variables

Set the following environment variables on your Orbit AI server:

```env
DATABASE_PROVIDER=supabase
STORAGE_PROVIDER=supabase
VECTOR_PROVIDER=supabase

SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_ANON_KEY=eyJ...
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_STORAGE_BUCKET=knowledge
SUPABASE_VECTOR_RPC=match_knowledge_vectors
SUPABASE_VECTOR_DIMENSIONS=128
EMBEDDING_DIMENSIONS=128
```

---

## Step 5: Verify Readiness

Start Orbit AI and query `GET /ready`. You should receive:

```json
{
  "status": "ready",
  "checks": {
    "database": true,
    "storage": true,
    "vector": true,
    "embedding_dimensions_match": true
  }
}
```
