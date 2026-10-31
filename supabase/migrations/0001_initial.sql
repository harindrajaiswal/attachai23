-- ============================================================================
-- ORBIT AI — COMPLETE INITIAL SUPABASE MIGRATION (0001_initial.sql)
-- ============================================================================
-- Execution instructions:
--   1. Open Supabase Dashboard -> SQL Editor -> New Query
--   2. Paste the entire contents of this file and click "Run"
--
-- Dimension Contract:
--   Default local embedding output = 128 dimensions
--   SUPABASE_VECTOR_DIMENSIONS     = 128
--   PostgreSQL vector column       = vector(128)
--   RPC query_embedding            = vector(128)
-- ============================================================================

-- 1. REQUIRED EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 2. UPDATED_AT TRIGGER FUNCTION
CREATE OR REPLACE FUNCTION public.orbit_set_updated_at()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.updated_at = timezone('utc', now());
    RETURN NEW;
END;
$$;

-- ============================================================================
-- 3. CORE TABLES (14 REQUIRED TABLES)
-- ============================================================================

-- 3.1 clients (Tenants)
CREATE TABLE IF NOT EXISTS public.clients (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    business_name TEXT NOT NULL,
    slug TEXT NOT NULL UNIQUE,
    company_name TEXT NOT NULL,
    assistant_name TEXT NOT NULL DEFAULT 'AI Assistant',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    system_prompt TEXT NOT NULL DEFAULT 'You are a helpful, accurate customer support assistant. Answer strictly using the provided knowledge base.',
    fallback_answer TEXT NOT NULL DEFAULT 'I do not have enough verified information in our knowledge base to answer that accurately. Please leave your contact details and our team will assist you.',
    welcome_message TEXT NOT NULL DEFAULT 'Hello! How can I help you today?',
    language TEXT NOT NULL DEFAULT 'en',
    model_override TEXT,
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.2,
    top_k INTEGER NOT NULL DEFAULT 5,
    min_similarity DOUBLE PRECISION NOT NULL DEFAULT 0.25,
    lead_capture_enabled BOOLEAN NOT NULL DEFAULT TRUE,
    lead_settings JSONB NOT NULL DEFAULT '{"require_email": true, "require_phone": false, "custom_prompt": "Leave your contact details so our team can follow up."}'::jsonb,
    notification_settings JSONB NOT NULL DEFAULT '{"email_enabled": false, "notify_emails": [], "webhook_url": null}'::jsonb,
    widget_config JSONB NOT NULL DEFAULT '{
        "primary_color": "#2563eb",
        "theme_mode": "light",
        "position": "bottom-right",
        "width": 380,
        "height": 560,
        "border_radius": 14,
        "logo_url": "",
        "avatar_url": "",
        "launcher_icon": "chat",
        "input_placeholder": "Ask a question...",
        "show_branding": true,
        "suggested_questions": ["What services do you offer?", "What are your pricing and hours?", "How can I contact support?"]
    }'::jsonb,
    analytics_settings JSONB NOT NULL DEFAULT '{"retention_days": 90, "track_latency": true}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

DROP TRIGGER IF EXISTS trg_clients_updated_at ON public.clients;
CREATE TRIGGER trg_clients_updated_at
BEFORE UPDATE ON public.clients
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- 3.2 api_keys (Hashed Tenant API Keys)
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    key_prefix TEXT NOT NULL,
    key_hash TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL DEFAULT 'Default Widget Key',
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    revoked_at TIMESTAMPTZ
);

-- 3.3 domains (Tenant Domain Allowlist)
CREATE TABLE IF NOT EXISTS public.domains (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    domain TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    UNIQUE (tenant_id, domain)
);

-- 3.4 prompt_versions (Versioned System Prompts per Tenant)
CREATE TABLE IF NOT EXISTS public.prompt_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    system_prompt TEXT NOT NULL,
    fallback_answer TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    created_by TEXT NOT NULL DEFAULT 'admin',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    UNIQUE (tenant_id, version_number)
);

-- 3.5 conversations (Tenant-Scoped Customer Sessions)
CREATE TABLE IF NOT EXISTS public.conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    summary TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

DROP TRIGGER IF EXISTS trg_conversations_updated_at ON public.conversations;
CREATE TRIGGER trg_conversations_updated_at
BEFORE UPDATE ON public.conversations
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- 3.6 messages (User & Assistant Messages)
CREATE TABLE IF NOT EXISTS public.messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    support_level TEXT NOT NULL DEFAULT 'UNKNOWN',
    grounding_passed BOOLEAN NOT NULL DEFAULT TRUE,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- 3.7 leads (Customer Lead Records with Idempotency Key)
CREATE TABLE IF NOT EXISTS public.leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    conversation_id UUID REFERENCES public.conversations(id) ON DELETE SET NULL,
    idempotency_key TEXT NOT NULL,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    phone TEXT NOT NULL DEFAULT '',
    requirement TEXT NOT NULL DEFAULT '',
    message TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'contacted', 'qualified', 'won', 'lost')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    UNIQUE (tenant_id, idempotency_key)
);

DROP TRIGGER IF EXISTS trg_leads_updated_at ON public.leads;
CREATE TRIGGER trg_leads_updated_at
BEFORE UPDATE ON public.leads
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- 3.8 lead_workflow (Lead Assignment, Status History & Internal Notes)
CREATE TABLE IF NOT EXISTS public.lead_workflow (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    lead_id UUID NOT NULL REFERENCES public.leads(id) ON DELETE CASCADE UNIQUE,
    status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'contacted', 'qualified', 'won', 'lost')),
    assignee TEXT NOT NULL DEFAULT '',
    internal_notes TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

DROP TRIGGER IF EXISTS trg_lead_workflow_updated_at ON public.lead_workflow;
CREATE TRIGGER trg_lead_workflow_updated_at
BEFORE UPDATE ON public.lead_workflow
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- 3.9 usage_logs (Tenant Usage, Request Counts & Latency Telemetry)
CREATE TABLE IF NOT EXISTS public.usage_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    operation TEXT NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 1,
    usage_units INTEGER NOT NULL DEFAULT 1,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    support_level TEXT NOT NULL DEFAULT 'UNKNOWN',
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- 3.10 audit_logs (Admin & Security Audit Trail)
CREATE TABLE IF NOT EXISTS public.audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES public.clients(id) ON DELETE SET NULL,
    actor TEXT NOT NULL DEFAULT 'admin',
    action TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- 3.11 knowledge_files (Uploaded Knowledge Document Metadata & Active Index Version)
CREATE TABLE IF NOT EXISTS public.knowledge_files (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    original_filename TEXT NOT NULL,
    safe_filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'processing' CHECK (status IN ('processing', 'ready', 'failed')),
    active_version INTEGER NOT NULL DEFAULT 1,
    chunk_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

DROP TRIGGER IF EXISTS trg_knowledge_files_updated_at ON public.knowledge_files;
CREATE TRIGGER trg_knowledge_files_updated_at
BEFORE UPDATE ON public.knowledge_files
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- 3.12 knowledge_vectors (128-Dimensional pgvector Embeddings with Versioned Activation)
CREATE TABLE IF NOT EXISTS public.knowledge_vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    source_id UUID NOT NULL REFERENCES public.knowledge_files(id) ON DELETE CASCADE,
    source_filename TEXT NOT NULL,
    source_version INTEGER NOT NULL DEFAULT 1,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(128) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- 3.13 settings (Safe Non-Secret Application Settings)
CREATE TABLE IF NOT EXISTS public.settings (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

-- 3.14 notifications (Persistent Outbox / Notification Delivery Records)
CREATE TABLE IF NOT EXISTS public.notifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    lead_id UUID REFERENCES public.leads(id) ON DELETE SET NULL,
    channel TEXT NOT NULL DEFAULT 'outbox',
    recipient TEXT NOT NULL DEFAULT '',
    subject TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now())
);

DROP TRIGGER IF EXISTS trg_notifications_updated_at ON public.notifications;
CREATE TRIGGER trg_notifications_updated_at
BEFORE UPDATE ON public.notifications
FOR EACH ROW EXECUTE FUNCTION public.orbit_set_updated_at();

-- ============================================================================
-- 4. INDEXES FOR PERFORMANCE & TENANT SCOPING
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_clients_slug ON public.clients(slug);
CREATE INDEX IF NOT EXISTS idx_clients_enabled ON public.clients(enabled);

CREATE INDEX IF NOT EXISTS idx_api_keys_tenant_id ON public.api_keys(tenant_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_key_hash ON public.api_keys(key_hash);

CREATE INDEX IF NOT EXISTS idx_domains_tenant_id ON public.domains(tenant_id);
CREATE INDEX IF NOT EXISTS idx_domains_domain ON public.domains(domain);

CREATE INDEX IF NOT EXISTS idx_prompt_versions_tenant_id ON public.prompt_versions(tenant_id, version_number DESC);

CREATE INDEX IF NOT EXISTS idx_conversations_tenant_id ON public.conversations(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON public.conversations(tenant_id, session_id);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON public.messages(conversation_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_messages_tenant_created ON public.messages(tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_leads_tenant_created ON public.leads(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_leads_status ON public.leads(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_lead_workflow_tenant_lead ON public.lead_workflow(tenant_id, lead_id);

CREATE INDEX IF NOT EXISTS idx_usage_logs_tenant_created ON public.usage_logs(tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created ON public.audit_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant ON public.audit_logs(tenant_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_knowledge_files_tenant_status ON public.knowledge_files(tenant_id, status);

CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_tenant_active ON public.knowledge_vectors(tenant_id, is_active, source_id, source_version);
CREATE INDEX IF NOT EXISTS idx_knowledge_vectors_embedding_hnsw ON public.knowledge_vectors USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_notifications_pending ON public.notifications(status, attempts, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_notifications_tenant ON public.notifications(tenant_id, created_at DESC);

-- ============================================================================
-- 5. VECTOR SIMILARITY SEARCH RPC (match_knowledge_vectors)
-- ============================================================================
-- Strictly enforces:
--   - 128-dimensional query_embedding
--   - Tenant ID isolation (kv.tenant_id = match_tenant_id)
--   - Active index version only (kv.is_active = TRUE AND kf.status = 'ready' AND kf.active_version = kv.source_version)
--   - Similarity threshold filtering and descending order
-- ============================================================================

CREATE OR REPLACE FUNCTION public.match_knowledge_vectors(
    query_embedding vector(128),
    match_tenant_id UUID,
    match_threshold DOUBLE PRECISION DEFAULT 0.20,
    match_count INTEGER DEFAULT 8
)
RETURNS TABLE (
    id UUID,
    tenant_id UUID,
    source_id UUID,
    source_filename TEXT,
    source_version INTEGER,
    chunk_index INTEGER,
    content TEXT,
    similarity DOUBLE PRECISION
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        kv.id,
        kv.tenant_id,
        kv.source_id,
        kv.source_filename,
        kv.source_version,
        kv.chunk_index,
        kv.content,
        (1.0 - (kv.embedding <=> query_embedding))::double precision AS similarity
    FROM public.knowledge_vectors AS kv
    INNER JOIN public.knowledge_files AS kf
        ON kf.id = kv.source_id
        AND kf.tenant_id = kv.tenant_id
    WHERE kv.tenant_id = match_tenant_id
      AND kv.is_active = TRUE
      AND kf.status = 'ready'
      AND kv.source_version = kf.active_version
      AND (1.0 - (kv.embedding <=> query_embedding)) >= match_threshold
    ORDER BY kv.embedding <=> query_embedding ASC
    LIMIT GREATEST(match_count, 1);
$$;

-- ============================================================================
-- 6. ROW LEVEL SECURITY (RLS) & TENANT ISOLATION POLICIES
-- ============================================================================
-- Documented tenant claim: (auth.jwt() ->> 'tenant_id')::uuid
-- Note: Backend service-role operations also explicitly filter every query by tenant_id.
-- Anonymous callers have zero direct table access.
-- ============================================================================

ALTER TABLE public.clients ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.domains ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.prompt_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.messages ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.leads ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.lead_workflow ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.usage_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.knowledge_files ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.knowledge_vectors ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.notifications ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    -- clients policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'clients' AND policyname = 'tenant_isolation_clients') THEN
        CREATE POLICY tenant_isolation_clients ON public.clients
            FOR ALL TO authenticated
            USING (id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- api_keys policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'api_keys' AND policyname = 'tenant_isolation_api_keys') THEN
        CREATE POLICY tenant_isolation_api_keys ON public.api_keys
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- domains policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'domains' AND policyname = 'tenant_isolation_domains') THEN
        CREATE POLICY tenant_isolation_domains ON public.domains
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- prompt_versions policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'prompt_versions' AND policyname = 'tenant_isolation_prompt_versions') THEN
        CREATE POLICY tenant_isolation_prompt_versions ON public.prompt_versions
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- conversations policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'conversations' AND policyname = 'tenant_isolation_conversations') THEN
        CREATE POLICY tenant_isolation_conversations ON public.conversations
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- messages policy (includes conversation ownership check)
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'messages' AND policyname = 'tenant_isolation_messages') THEN
        CREATE POLICY tenant_isolation_messages ON public.messages
            FOR ALL TO authenticated
            USING (
                tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid
                AND EXISTS (
                    SELECT 1 FROM public.conversations c
                    WHERE c.id = messages.conversation_id
                      AND c.tenant_id = messages.tenant_id
                )
            )
            WITH CHECK (
                tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid
                AND EXISTS (
                    SELECT 1 FROM public.conversations c
                    WHERE c.id = messages.conversation_id
                      AND c.tenant_id = messages.tenant_id
                )
            );
    END IF;

    -- leads policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'leads' AND policyname = 'tenant_isolation_leads') THEN
        CREATE POLICY tenant_isolation_leads ON public.leads
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- lead_workflow policy (includes lead ownership check)
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'lead_workflow' AND policyname = 'tenant_isolation_lead_workflow') THEN
        CREATE POLICY tenant_isolation_lead_workflow ON public.lead_workflow
            FOR ALL TO authenticated
            USING (
                tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid
                AND EXISTS (
                    SELECT 1 FROM public.leads l
                    WHERE l.id = lead_workflow.lead_id
                      AND l.tenant_id = lead_workflow.tenant_id
                )
            )
            WITH CHECK (
                tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid
                AND EXISTS (
                    SELECT 1 FROM public.leads l
                    WHERE l.id = lead_workflow.lead_id
                      AND l.tenant_id = lead_workflow.tenant_id
                )
            );
    END IF;

    -- usage_logs policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'usage_logs' AND policyname = 'tenant_isolation_usage_logs') THEN
        CREATE POLICY tenant_isolation_usage_logs ON public.usage_logs
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- audit_logs policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'audit_logs' AND policyname = 'tenant_isolation_audit_logs') THEN
        CREATE POLICY tenant_isolation_audit_logs ON public.audit_logs
            FOR SELECT TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- knowledge_files policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'knowledge_files' AND policyname = 'tenant_isolation_knowledge_files') THEN
        CREATE POLICY tenant_isolation_knowledge_files ON public.knowledge_files
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- knowledge_vectors policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'knowledge_vectors' AND policyname = 'tenant_isolation_knowledge_vectors') THEN
        CREATE POLICY tenant_isolation_knowledge_vectors ON public.knowledge_vectors
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;

    -- notifications policy
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname = 'public' AND tablename = 'notifications' AND policyname = 'tenant_isolation_notifications') THEN
        CREATE POLICY tenant_isolation_notifications ON public.notifications
            FOR ALL TO authenticated
            USING (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid)
            WITH CHECK (tenant_id = NULLIF(auth.jwt() ->> 'tenant_id', '')::uuid);
    END IF;
END
$$;

-- ============================================================================
-- 7. PRIVATE SUPABASE STORAGE BUCKET (knowledge)
-- ============================================================================
-- Creates the private 'knowledge' bucket idempotently when storage.buckets exists.
-- ============================================================================

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'storage' AND table_name = 'buckets'
    ) THEN
        INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
        VALUES (
            'knowledge',
            'knowledge',
            FALSE,
            10485760,
            ARRAY[
                'text/plain',
                'text/markdown',
                'text/csv',
                'application/pdf',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            ]
        )
        ON CONFLICT (id) DO UPDATE
        SET public = FALSE,
            file_size_limit = EXCLUDED.file_size_limit,
            allowed_mime_types = EXCLUDED.allowed_mime_types;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'storage' AND table_name = 'objects'
    ) THEN
        IF NOT EXISTS (
            SELECT 1 FROM pg_policies
            WHERE schemaname = 'storage'
              AND tablename = 'objects'
              AND policyname = 'tenant_isolation_knowledge_storage'
        ) THEN
            CREATE POLICY tenant_isolation_knowledge_storage ON storage.objects
                FOR ALL TO authenticated
                USING (
                    bucket_id = 'knowledge'
                    AND (storage.foldername(name))[1] = (auth.jwt() ->> 'tenant_id')
                )
                WITH CHECK (
                    bucket_id = 'knowledge'
                    AND (storage.foldername(name))[1] = (auth.jwt() ->> 'tenant_id')
                );
        END IF;
    END IF;
END
$$;
