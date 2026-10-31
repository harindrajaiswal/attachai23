"""CORS and security header policies for Orbit AI."""

from typing import Dict, List, Optional
from app.utils.validators import domain_matches, normalize_domain


def build_security_headers(is_production: bool = False) -> Dict[str, str]:
    """Return standard hardening headers for every HTTP response."""
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Cache-Control": "no-store",
    }
    if is_production:
        headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return headers


def resolve_cors_headers(
    origin: Optional[str],
    allowed_origins: List[str],
    is_public_widget_route: bool = False,
) -> Dict[str, str]:
    """Compute strict CORS headers for admin and public widget routes."""
    if not origin or not origin.strip():
        return {}

    origin_clean = origin.strip()
    if is_public_widget_route:
        # Public /v1/* widget routes validate tenant domain allowlists in authenticate_public_request;
        # echo the requesting Origin for browser preflight compatibility without using wildcard with credentials.
        return {
            "Access-Control-Allow-Origin": origin_clean,
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, x-api-key, x-client-id, x-idempotency-key",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }

    # For non-widget routes, strictly check global ALLOWED_ORIGINS
    allowed = False
    if "*" in allowed_origins or origin_clean in allowed_origins:
        allowed = True
    else:
        try:
            origin_host = normalize_domain(origin_clean)
            for item in allowed_origins:
                try:
                    if domain_matches(origin_host, normalize_domain(item)):
                        allowed = True
                        break
                except Exception:
                    continue
        except Exception:
            allowed = False

    if not allowed:
        return {}

    return {
        "Access-Control-Allow-Origin": origin_clean,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, x-admin-api-key, x-csrf-token, x-api-key",
        "Vary": "Origin",
    }
