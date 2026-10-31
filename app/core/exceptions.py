"""Domain exceptions and structured HTTP error mapping for Orbit AI."""

from typing import Any, Dict, Optional


class OrbitError(Exception):
    """Base application exception with safe public message and HTTP status code."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        code: str = "ORBIT_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class ConfigurationError(OrbitError):
    """Raised when startup or provider configuration is invalid."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            code="CONFIGURATION_ERROR",
            details=details,
        )


class AuthenticationError(OrbitError):
    """Raised when admin or tenant credentials are missing or invalid."""

    def __init__(self, message: str = "Authentication failed.") -> None:
        super().__init__(
            message=message,
            status_code=401,
            code="AUTHENTICATION_FAILED",
        )


class AuthorizationError(OrbitError):
    """Raised when a caller attempts unauthorized or cross-tenant access."""

    def __init__(self, message: str = "Access denied for this tenant or origin.") -> None:
        super().__init__(
            message=message,
            status_code=403,
            code="AUTHORIZATION_DENIED",
        )


class NotFoundError(OrbitError):
    """Raised when a requested tenant-scoped resource does not exist."""

    def __init__(self, message: str = "Resource not found.") -> None:
        super().__init__(
            message=message,
            status_code=404,
            code="NOT_FOUND",
        )


class ValidationError(OrbitError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=422,
            code="VALIDATION_ERROR",
            details=details,
        )


class RateLimitExceededError(OrbitError):
    """Raised when per-IP or per-tenant rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded. Please retry shortly.", retry_after: int = 60) -> None:
        super().__init__(
            message=message,
            status_code=429,
            code="RATE_LIMIT_EXCEEDED",
            details={"retry_after": retry_after},
        )


class ProviderError(OrbitError):
    """Raised when an external LLM, embedding, storage, or database provider fails."""

    def __init__(
        self,
        message: str = "Upstream provider operation failed.",
        status_code: int = 502,
        provider: str = "unknown",
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            code="PROVIDER_ERROR",
            details={"provider": provider},
        )
