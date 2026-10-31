-- ============================================================================
-- ORBIT AI — MIGRATION 0002: MESSAGE FEEDBACK & CONVERSATION MEMORY INDEXES
-- Idempotent upgrade migration for existing and new Supabase deployments.
-- Execute in the Supabase SQL Editor after 0001_initial.sql.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- 1. Message Feedback Table (Supports Like/Dislike, reason, and duplicate prevention)
CREATE TABLE IF NOT EXISTS public.message_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES public.clients(id) ON DELETE CASCADE,
    conversation_id UUID NOT NULL REFERENCES public.conversations(id) ON DELETE CASCADE,
    message_id UUID NOT NULL REFERENCES public.messages(id) ON DELETE CASCADE,
    feedback_type TEXT NOT NULL CHECK (feedback_type IN ('like', 'dislike')),
    reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT timezone('utc', now()),
    CONSTRAINT uq_message_feedback_tenant_message UNIQUE (tenant_id, message_id)
);

-- 2. Indexes for fast tenant-scoped feedback lookups
CREATE INDEX IF NOT EXISTS idx_message_feedback_tenant_id
    ON public.message_feedback(tenant_id);

CREATE INDEX IF NOT EXISTS idx_message_feedback_conversation_id
    ON public.message_feedback(tenant_id, conversation_id);

CREATE INDEX IF NOT EXISTS idx_message_feedback_message_id
    ON public.message_feedback(tenant_id, message_id);

-- 3. Automatic updated_at trigger
DROP TRIGGER IF EXISTS trg_message_feedback_updated_at ON public.message_feedback;
CREATE TRIGGER trg_message_feedback_updated_at
BEFORE UPDATE ON public.message_feedback
FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- 4. Row Level Security (RLS)
ALTER TABLE public.message_feedback ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all_message_feedback ON public.message_feedback;
CREATE POLICY service_role_all_message_feedback
    ON public.message_feedback
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);
