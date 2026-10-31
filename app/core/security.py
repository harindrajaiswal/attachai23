"""Cryptographic utilities, API key hashing, constant-time comparison, and sessions."""

import hashlib
import hmac
import secrets
import time
from typing import Dict, Optional, Tuple


def constant_time_equals(val_a: str, val_b: str) -> bool:
    """Compare two strings in constant time to prevent timing side-channel attacks."""
    if not isinstance(val_a, str) or not isinstance(val_b, str):
        return False
    a_bytes = val_a.encode("utf-8")
    b_bytes = val_b.encode("utf-8")
    return hmac.compare_digest(a_bytes, b_bytes)


def hash_api_key(raw_key: str) -> str:
    """Compute a deterministic SHA-256 hex digest of a tenant API key."""
    cleaned = (raw_key or "").strip()
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def generate_tenant_api_key() -> Tuple[str, str, str]:
    """Generate a cryptographically random tenant API key.

    Returns:
        Tuple of (raw_key, key_prefix, key_hash)
    """
    token = secrets.token_urlsafe(24).replace("-", "").replace("_", "")[:32]
    raw_key = f"orbit_live_{token}"
    key_prefix = raw_key[:16]
    key_hash = hash_api_key(raw_key)
    return raw_key, key_prefix, key_hash


class AdminSessionManager:
    """Server-side signed session and CSRF manager for the Admin Dashboard."""

    def __init__(self, secret_key: str, ttl_seconds: int = 28800) -> None:
        self._secret = (secret_key or "orbit-fallback-secret").encode("utf-8")
        self._ttl = ttl_seconds
        self._sessions: Dict[str, Tuple[float, str]] = {}

    def create_session(self) -> Tuple[str, str]:
        """Create an authenticated admin session token and paired CSRF token."""
        session_id = secrets.token_urlsafe(32)
        csrf_token = secrets.token_urlsafe(24)
        expires_at = time.time() + self._ttl
        signature = hmac.new(
            self._secret, f"{session_id}:{csrf_token}:{int(expires_at)}".encode("utf-8"), hashlib.sha256
        ).hexdigest()
        signed_cookie = f"{session_id}.{int(expires_at)}.{signature}"
        self._sessions[session_id] = (expires_at, csrf_token)
        return signed_cookie, csrf_token

    def verify_session(self, cookie_value: Optional[str]) -> Optional[str]:
        """Verify session cookie and return its CSRF token if valid, else None."""
        if not cookie_value or "." not in cookie_value:
            return None
        parts = cookie_value.split(".")
        if len(parts) != 3:
            return None
        session_id, exp_str, sig = parts
        record = self._sessions.get(session_id)
        if not record:
            return None
        expires_at, csrf_token = record
        if time.time() > expires_at:
            self._sessions.pop(session_id, None)
            return None
        expected_sig = hmac.new(
            self._secret, f"{session_id}:{csrf_token}:{exp_str}".encode("utf-8"), hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return None
        return csrf_token

    def verify_csrf(self, cookie_value: Optional[str], provided_csrf: Optional[str]) -> bool:
        """Verify that provided_csrf matches the active session's CSRF token."""
        expected_csrf = self.verify_session(cookie_value)
        if not expected_csrf or not provided_csrf:
            return False
        return constant_time_equals(expected_csrf, provided_csrf.strip())

    def destroy_session(self, cookie_value: Optional[str]) -> None:
        """Invalidate an admin session."""
        if not cookie_value or "." not in cookie_value:
            return
        session_id = cookie_value.split(".")[0]
        self._sessions.pop(session_id, None)
