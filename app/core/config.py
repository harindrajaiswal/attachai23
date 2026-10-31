"""Application configuration loader with startup validation.

Supports Pydantic Settings (v2) in standard environments and a deterministic
environment loader fallback for zero-dependency test execution.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.core.exceptions import ConfigurationError


def _parse_bool(val: Any, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return default


def _parse_int(val: Any, default: int) -> int:
    if val is None or str(val).strip() == "":
        return default
    try:
        return int(str(val).strip())
    except ValueError as exc:
        raise ConfigurationError(f"Expected integer configuration value, got '{val}'.") from exc


def _parse_float(val: Any, default: float) -> float:
    if val is None or str(val).strip() == "":
        return default
    try:
        return float(str(val).strip())
    except ValueError as exc:
        raise ConfigurationError(f"Expected float configuration value, got '{val}'.") from exc


class Settings:
    """Centralized validated configuration for Orbit AI."""

    def __init__(self, env_overrides: Optional[Dict[str, Any]] = None) -> None:
        env: Dict[str, Any] = dict(os.environ)
        if env_overrides:
            for k, v in env_overrides.items():
                if v is None:
                    env.pop(k, None)
                else:
                    env[k] = str(v)

        # 1. Core Application Environment (APP_ENV takes precedence over legacy NODE_ENV)
        raw_app_env = env.get("APP_ENV") or env.get("NODE_ENV") or "development"
        self.app_env: str = raw_app_env.strip().lower()
        if self.app_env not in ("development", "test", "production", "staging"):
            raise ConfigurationError(f"Invalid APP_ENV '{self.app_env}'. Allowed: development, test, staging, production.")

        self.host: str = (env.get("HOST") or "0.0.0.0").strip()
        self.port: int = _parse_int(env.get("PORT"), 8000)
        self.public_base_url: str = (env.get("PUBLIC_BASE_URL") or f"http://localhost:{self.port}").strip().rstrip("/")

        # Admin Authentication
        self.admin_api_key: str = (env.get("ADMIN_API_KEY") or "").strip()
        if not self.admin_api_key and self.app_env in ("development", "test"):
            self.admin_api_key = "orbit-dev-admin-key-2026"

        # ALLOWED_ORIGINS takes precedence over legacy CORS_ORIGINS
        raw_origins = env.get("ALLOWED_ORIGINS")
        if raw_origins is None or raw_origins.strip() == "":
            raw_origins = env.get("CORS_ORIGINS") or "http://localhost:8000,http://localhost:3000"
        self.allowed_origins: List[str] = [
            o.strip() for o in raw_origins.split(",") if o.strip()
        ]

        # 2. Provider Selections
        self.database_provider: str = (env.get("DATABASE_PROVIDER") or "sqlite").strip().lower()
        self.storage_provider: str = (env.get("STORAGE_PROVIDER") or "local").strip().lower()
        self.vector_provider: str = (env.get("VECTOR_PROVIDER") or "local").strip().lower()
        self.notification_provider: str = (
            env.get("NOTIFICATION_PROVIDER")
            or ("supabase" if self.app_env == "production" else "outbox")
        ).strip().lower()

        # SQLite & Local paths
        self.sqlite_db_path: str = (env.get("SQLITE_DB_PATH") or "data/orbit_local.db").strip()
        self.local_storage_dir: str = (env.get("LOCAL_STORAGE_DIR") or "local_storage").strip()

        # 3. Supabase Settings
        self.supabase_url: str = (env.get("SUPABASE_URL") or "").strip().rstrip("/")
        self.supabase_service_role_key: str = (env.get("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
        self.supabase_storage_bucket: str = (env.get("SUPABASE_STORAGE_BUCKET") or "knowledge").strip()
        self.supabase_vector_dimensions: int = _parse_int(env.get("SUPABASE_VECTOR_DIMENSIONS"), 128)

        # Optional Qdrant Settings
        self.qdrant_url: str = (env.get("QDRANT_URL") or "").strip().rstrip("/")
        self.qdrant_api_key: str = (env.get("QDRANT_API_KEY") or "").strip()
        self.qdrant_collection: str = (env.get("QDRANT_COLLECTION") or "orbit_knowledge").strip()

        # 4. LLM Settings
        self.llm_provider_type: str = (env.get("LLM_PROVIDER_TYPE") or "local").strip().lower()
        self.llm_provider_name: str = (env.get("LLM_PROVIDER_NAME") or self.llm_provider_type).strip()
        self.llm_base_url: str = (env.get("LLM_BASE_URL") or "").strip().rstrip("/")
        self.llm_api_key_env: str = (env.get("LLM_API_KEY_ENV") or "").strip()
        resolved_llm_key = env.get("LLM_API_KEY") or ""
        if not resolved_llm_key and self.llm_api_key_env:
            resolved_llm_key = env.get(self.llm_api_key_env) or ""
        self.llm_api_key: str = resolved_llm_key.strip()
        self.llm_model: str = (env.get("LLM_MODEL") or "local-rag-synthesizer").strip()
        self.llm_timeout_ms: int = _parse_int(env.get("LLM_TIMEOUT_MS"), 15000)
        self.llm_max_retries: int = _parse_int(env.get("LLM_MAX_RETRIES"), 2)
        self.llm_temperature: float = _parse_float(env.get("LLM_TEMPERATURE"), 0.2)
        self.llm_max_tokens: int = _parse_int(env.get("LLM_MAX_TOKENS"), 800)
        self.llm_stream: bool = _parse_bool(env.get("LLM_STREAM"), False)
        self.llm_chat_path: str = (env.get("LLM_CHAT_PATH") or "/chat/completions").strip()
        self.llm_health_path: str = (env.get("LLM_HEALTH_PATH") or "/models").strip()
        self.llm_auth_header_name: str = (env.get("LLM_AUTH_HEADER_NAME") or "Authorization").strip()
        self.llm_auth_scheme: str = (env.get("LLM_AUTH_SCHEME") or "Bearer").strip()

        raw_headers = (env.get("LLM_HEADERS") or "").strip()
        self.llm_headers: Dict[str, str] = {}
        if raw_headers:
            try:
                parsed_h = json.loads(raw_headers)
                if isinstance(parsed_h, dict):
                    self.llm_headers = {str(k): str(v) for k, v in parsed_h.items()}
                else:
                    raise ConfigurationError("LLM_HEADERS must be a JSON object.")
            except json.JSONDecodeError as exc:
                raise ConfigurationError("Invalid JSON in LLM_HEADERS.") from exc

        raw_params = (env.get("LLM_QUERY_PARAMS") or "").strip()
        self.llm_query_params: Dict[str, str] = {}
        if raw_params:
            try:
                parsed_p = json.loads(raw_params)
                if isinstance(parsed_p, dict):
                    self.llm_query_params = {str(k): str(v) for k, v in parsed_p.items()}
                else:
                    raise ConfigurationError("LLM_QUERY_PARAMS must be a JSON object.")
            except json.JSONDecodeError as exc:
                raise ConfigurationError("Invalid JSON in LLM_QUERY_PARAMS.") from exc

        # 5. Embeddings Settings
        self.embedding_provider: str = (env.get("EMBEDDING_PROVIDER") or "local").strip().lower()
        self.embedding_dimensions: int = _parse_int(env.get("EMBEDDING_DIMENSIONS"), 128)
        self.embedding_base_url: str = (env.get("EMBEDDING_BASE_URL") or "").strip().rstrip("/")
        self.embedding_model: str = (env.get("EMBEDDING_MODEL") or "orbit-local-128d").strip()
        self.embedding_api_key: str = (env.get("EMBEDDING_API_KEY") or "").strip()

        # 6. SMTP Settings (when NOTIFICATION_PROVIDER=smtp)
        self.smtp_host: str = (env.get("SMTP_HOST") or "").strip()
        self.smtp_port: int = _parse_int(env.get("SMTP_PORT"), 587)
        self.smtp_user: str = (env.get("SMTP_USER") or "").strip()
        self.smtp_pass: str = (env.get("SMTP_PASS") or "").strip()
        self.smtp_from: str = (env.get("SMTP_FROM") or "notifications@orbit-ai.local").strip()

        # 7. Operational & RAG Limits
        self.max_upload_size_mb: int = _parse_int(env.get("MAX_UPLOAD_SIZE_MB"), 10)
        self.chunk_size: int = _parse_int(env.get("CHUNK_SIZE"), 900)
        self.chunk_overlap: int = _parse_int(env.get("CHUNK_OVERLAP"), 120)
        self.default_top_k: int = _parse_int(env.get("DEFAULT_TOP_K"), 5)
        self.default_min_similarity: float = _parse_float(env.get("DEFAULT_MIN_SIMILARITY"), 0.25)
        self.max_message_length: int = _parse_int(env.get("MAX_MESSAGE_LENGTH"), 4000)
        self.conversation_history_limit: int = _parse_int(env.get("CONVERSATION_HISTORY_LIMIT"), 10)
        self.public_rate_limit_per_minute: int = _parse_int(env.get("PUBLIC_RATE_LIMIT_PER_MINUTE"), 60)
        self.admin_rate_limit_per_minute: int = _parse_int(env.get("ADMIN_RATE_LIMIT_PER_MINUTE"), 120)

        # 8. RAG Diagnostics & Legacy Debug Flags (strictly off by default)
        self.orbit_rag_diagnostics: bool = _parse_bool(env.get("ORBIT_RAG_DIAGNOSTICS"), False)
        self.orbit_temp_debug_client_id: str = (env.get("ORBIT_TEMP_DEBUG_CLIENT_ID") or "").strip()
        self.orbit_temp_debug_gousto: bool = _parse_bool(env.get("ORBIT_TEMP_DEBUG_GOUSTO"), False)
        self.orbit_temp_debug_llm_input: bool = _parse_bool(env.get("ORBIT_TEMP_DEBUG_LLM_INPUT"), False)
        self.orbit_temp_debug_puregym: bool = _parse_bool(env.get("ORBIT_TEMP_DEBUG_PUREGYM"), False)

        # 9. Optional TTS Settings (server-only; stays disabled unless full config supplied)
        self.tts_api_key: str = (env.get("TTS_API_KEY") or "").strip()
        self.tts_provider: str = (env.get("TTS_PROVIDER") or "").strip().lower()
        self.tts_base_url: str = (env.get("TTS_BASE_URL") or "").strip()
        self.tts_model: str = (env.get("TTS_MODEL") or "").strip()
        self.tts_voice: str = (env.get("TTS_VOICE") or "").strip()

        self.validate()

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def tts_enabled(self) -> bool:
        """TTS is only enabled when key, provider, base URL, and model/voice are all configured."""
        return bool(
            self.tts_api_key
            and self.tts_provider
            and self.tts_base_url
            and (self.tts_model or self.tts_voice)
        )

    def known_secrets(self) -> List[str]:
        """Return list of configured secrets for log/error redaction."""
        return [
            s
            for s in (
                self.admin_api_key,
                self.supabase_service_role_key,
                self.llm_api_key,
                self.embedding_api_key,
                self.tts_api_key,
                self.smtp_pass,
                self.qdrant_api_key,
            )
            if s
        ]

    def validate(self) -> None:
        """Validate provider compatibility, credentials, and vector dimensions at startup."""
        if self.database_provider not in ("sqlite", "supabase"):
            raise ConfigurationError(
                f"Invalid DATABASE_PROVIDER '{self.database_provider}'. Must be 'supabase' or 'sqlite'."
            )
        if self.storage_provider not in ("local", "supabase"):
            raise ConfigurationError(
                f"Invalid STORAGE_PROVIDER '{self.storage_provider}'. Must be 'supabase' or 'local'."
            )
        if self.vector_provider not in ("local", "supabase", "qdrant"):
            raise ConfigurationError(
                f"Invalid VECTOR_PROVIDER '{self.vector_provider}'. Must be 'supabase', 'local', or 'qdrant'."
            )
        if self.notification_provider not in ("outbox", "supabase", "smtp"):
            raise ConfigurationError(
                f"Invalid NOTIFICATION_PROVIDER '{self.notification_provider}'. Must be 'supabase', 'smtp', or 'outbox'."
            )
        if self.llm_provider_type not in ("local", "openai-compatible"):
            raise ConfigurationError(
                f"Invalid LLM_PROVIDER_TYPE '{self.llm_provider_type}'. Must be 'local' or 'openai-compatible'."
            )
        if self.embedding_provider not in ("local", "openai-compatible"):
            raise ConfigurationError(
                f"Invalid EMBEDDING_PROVIDER '{self.embedding_provider}'. Must be 'local' or 'openai-compatible'."
            )

        # Enforce strict vector dimension contract
        if self.embedding_dimensions <= 0 or self.supabase_vector_dimensions <= 0:
            raise ConfigurationError("Vector dimensions must be positive integers.")
        if self.embedding_dimensions != self.supabase_vector_dimensions:
            raise ConfigurationError(
                f"Vector dimension mismatch: EMBEDDING_DIMENSIONS ({self.embedding_dimensions}) "
                f"must match SUPABASE_VECTOR_DIMENSIONS ({self.supabase_vector_dimensions})."
            )
        if self.embedding_provider == "local" and self.embedding_dimensions != 128:
            raise ConfigurationError(
                f"Local deterministic embedding provider requires 128 dimensions, got {self.embedding_dimensions}."
            )

        # Production security requirements
        if self.is_production:
            if not self.admin_api_key or len(self.admin_api_key) < 12 or self.admin_api_key == "change-me-in-production-min-16-chars":
                raise ConfigurationError(
                    "Production environment requires a strong ADMIN_API_KEY (minimum 12 characters)."
                )

        # Supabase provider requirements
        uses_supabase = (
            self.database_provider == "supabase"
            or self.storage_provider == "supabase"
            or self.vector_provider == "supabase"
            or self.notification_provider == "supabase"
        )
        if uses_supabase:
            if not self.supabase_url or not self.supabase_service_role_key:
                raise ConfigurationError(
                    "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required when any provider is set to 'supabase'."
                )
            if not self.supabase_storage_bucket:
                raise ConfigurationError("SUPABASE_STORAGE_BUCKET must not be empty.")

        # Qdrant provider requirements
        if self.vector_provider == "qdrant" and not self.qdrant_url:
            raise ConfigurationError("QDRANT_URL is required when VECTOR_PROVIDER='qdrant'.")

        # OpenAI-compatible LLM requirements
        if self.llm_provider_type == "openai-compatible":
            if not self.llm_base_url or not self.llm_model or not self.llm_api_key:
                raise ConfigurationError(
                    "LLM_BASE_URL, LLM_MODEL, and LLM_API_KEY are required when LLM_PROVIDER_TYPE='openai-compatible'."
                )

        # OpenAI-compatible Embedding requirements
        if self.embedding_provider == "openai-compatible":
            if not self.embedding_base_url or not self.embedding_model or not self.embedding_api_key:
                raise ConfigurationError(
                    "EMBEDDING_BASE_URL, EMBEDDING_MODEL, and EMBEDDING_API_KEY are required when EMBEDDING_PROVIDER='openai-compatible'."
                )

        # SMTP notification requirements
        if self.notification_provider == "smtp" and not self.smtp_host:
            raise ConfigurationError("SMTP_HOST is required when NOTIFICATION_PROVIDER='smtp'.")


def load_settings(env_overrides: Optional[Dict[str, Any]] = None) -> Settings:
    """Create and validate a Settings instance."""
    return Settings(env_overrides=env_overrides)
