"""Authentication helpers for admin and public tenant endpoints."""

from typing import Any, Dict, Optional
from app.core.dependencies import ServiceContainer
from app.core.exceptions import AuthenticationError, AuthorizationError
from app.core.security import constant_time_equals

ADMIN_SESSION_COOKIE = "orbit_admin_session"


def verify_admin_credentials(
    container: ServiceContainer,
    admin_header_key: Optional[str] = None,
    session_cookie: Optional[str] = None,
    csrf_header: Optional[str] = None,
    method: str = "GET",
    require_header_only: bool = False,
) -> str:
    """Authenticate an administrator request via x-admin-api-key or signed HttpOnly session cookie."""
    expected_key = container.settings.admin_api_key
    if admin_header_key is not None and admin_header_key.strip() != "":
        if constant_time_equals(admin_header_key.strip(), expected_key):
            return "api_key"
        raise AuthenticationError("Invalid x-admin-api-key credential.")

    if not require_header_only and session_cookie:
        csrf_token = container.session_manager.verify_session(session_cookie)
        if csrf_token:
            if method.upper() in ("POST", "PUT", "PATCH", "DELETE"):
                if not container.session_manager.verify_csrf(session_cookie, csrf_header):
                    raise AuthorizationError("CSRF token validation failed for session-authenticated request.")
            return "session"

    raise AuthenticationError("Missing or invalid administrator credentials (x-admin-api-key required).")


async def verify_public_tenant_request(
    container: ServiceContainer,
    api_key_header: Optional[str],
    origin_header: Optional[str] = None,
    client_id_header: Optional[str] = None,
    client_ip: str = "127.0.0.1",
) -> Dict[str, Any]:
    """Authenticate a public widget request and enforce per-IP & per-tenant rate limits."""
    container.rate_limiter.check(
        f"pub_ip:{client_ip}",
        container.settings.public_rate_limit_per_minute,
    )
    tenant = await container.client_service.authenticate_public_request(
        raw_api_key=api_key_header,
        origin_header=origin_header,
        expected_client_id=client_id_header,
    )
    container.rate_limiter.check(
        f"pub_tenant:{tenant['id']}",
        container.settings.public_rate_limit_per_minute * 2,
    )
    return tenant
