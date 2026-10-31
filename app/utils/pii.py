"""PII and secret redaction utilities.

Fixes original Orbit AI Defect #5:
Correctly detects and sanitizes email addresses, phone numbers, API tokens,
Bearer headers, Supabase service-role JWTs, and secret keys in strings and dicts.
"""

import re
from typing import Any, Dict, Iterable, List, Set

# Email pattern
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

# International & domestic phone numbers (requires at least 7 digits and standard separators)
PHONE_PATTERN = re.compile(
    r"(?:(?:\+|00)\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?|\d{2,4}[\s.-])?\d{3,4}[\s.-]\d{3,4}\b"
)

# API keys, Bearer tokens, JWTs, sk-*, orbit_*, sbp_*, eyJ*
TOKEN_PATTERN = re.compile(
    r"(?:Bearer\s+[A-Za-z0-9\-._~+/]+=*|"
    r"\bsk-[A-Za-z0-9\-_]{12,}\b|"
    r"\borbit_(?:live|test|key)_[A-Za-z0-9\-_]{10,}\b|"
    r"\bsbp_[A-Za-z0-9\-_]{12,}\b|"
    r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b)",
    re.IGNORECASE,
)

# Key-value assignment patterns in logs
KV_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|service[_-]?role[_-]?key|admin[_-]?api[_-]?key|authorization|password|secret|token)"
    r"(\s*[:=]\s*['\"]?)([^'\"\s,&]+)(['\"]?)"
)

SENSITIVE_DICT_KEYS: Set[str] = {
    "admin_api_key",
    "x-admin-api-key",
    "x-api-key",
    "api_key",
    "apikey",
    "key_hash",
    "raw_key",
    "supabase_service_role_key",
    "service_role_key",
    "llm_api_key",
    "embedding_api_key",
    "tts_api_key",
    "smtp_pass",
    "password",
    "secret",
    "authorization",
    "cookie",
    "set-cookie",
}


def redact_text(value: str, extra_secrets: Iterable[str] = ()) -> str:
    """Redact emails, phone numbers, API tokens, and known secret values from text."""
    if not value or not isinstance(value, str):
        return value

    redacted = value
    for secret in extra_secrets:
        if secret and len(secret.strip()) >= 6:
            redacted = redacted.replace(secret.strip(), "[REDACTED_SECRET]")

    redacted = KV_SECRET_PATTERN.sub(r"\1\2[REDACTED_SECRET]\4", redacted)
    redacted = TOKEN_PATTERN.sub("[REDACTED_TOKEN]", redacted)
    redacted = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", redacted)
    redacted = PHONE_PATTERN.sub("[REDACTED_PHONE]", redacted)
    return redacted


def redact_data(data: Any, extra_secrets: Iterable[str] = ()) -> Any:
    """Recursively redact sensitive keys and PII from dictionaries and lists."""
    if isinstance(data, str):
        return redact_text(data, extra_secrets=extra_secrets)
    if isinstance(data, dict):
        sanitized: Dict[str, Any] = {}
        for k, v in data.items():
            key_lower = str(k).lower().strip()
            if key_lower in SENSITIVE_DICT_KEYS or any(
                s in key_lower for s in ("secret", "password", "api_key", "service_role", "authorization")
            ):
                sanitized[k] = "[REDACTED_SECRET]"
            else:
                sanitized[k] = redact_data(v, extra_secrets=extra_secrets)
        return sanitized
    if isinstance(data, list):
        return [redact_data(item, extra_secrets=extra_secrets) for item in data]
    if isinstance(data, tuple):
        return tuple(redact_data(item, extra_secrets=extra_secrets) for item in data)
    return data


def mask_secret_preview(secret: str) -> str:
    """Return a safe masked preview of a secret without exposing its value."""
    if not secret:
        return "not-configured"
    cleaned = secret.strip()
    if len(cleaned) <= 8:
        return "****"
    return f"{cleaned[:4]}...{cleaned[-2:]} (masked)"
