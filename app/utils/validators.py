"""Input validators for filenames, domains, emails, slugs, colors, and storage paths.

Fixes Defect #13:
Filename normalization uses `<tenant-id>/<source-id>/<safe-filename>` so two files
with similar normalized names never collide in storage.
"""

import re
from urllib.parse import urlparse
from app.core.exceptions import ValidationError

SLUG_REGEX = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
HEX_COLOR_REGEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
EMAIL_VALIDATE_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
SAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

ALLOWED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "text/csv",
    "application/csv",
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream",
}


def normalize_slug(value: str) -> str:
    """Normalize and validate a tenant slug."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    if not cleaned or len(cleaned) < 2 or len(cleaned) > 64 or not SLUG_REGEX.match(cleaned):
        raise ValidationError("Invalid slug. Must contain lowercase letters, numbers, and hyphens (2-64 chars).")
    return cleaned


def validate_email_address(email: str) -> str:
    """Validate and normalize an email address."""
    cleaned = (email or "").strip().lower()
    if not cleaned or len(cleaned) > 254 or not EMAIL_VALIDATE_REGEX.match(cleaned):
        raise ValidationError("A valid email address is required.")
    return cleaned


def validate_hex_color(color: str) -> str:
    """Validate a CSS hex color string."""
    cleaned = (color or "").strip()
    if not HEX_COLOR_REGEX.match(cleaned):
        raise ValidationError(f"Invalid hex color '{color}'. Expected #RRGGBB or #RGB.")
    return cleaned.lower()


def normalize_domain(domain_or_origin: str) -> str:
    """Extract and normalize a hostname from a domain or Origin/Referer header."""
    raw = (domain_or_origin or "").strip().lower()
    if not raw:
        raise ValidationError("Domain cannot be empty.")
    if "://" in raw:
        parsed = urlparse(raw)
        host = parsed.hostname or ""
    else:
        host = raw.split("/")[0].split(":")[0]
    host = host.strip(".")
    if not host or " " in host or ("." not in host and host not in ("localhost", "127.0.0.1", "testserver")):
        raise ValidationError(f"Invalid domain '{domain_or_origin}'.")
    return host


def domain_matches(origin_host: str, allowed_domain: str) -> bool:
    """Check if origin_host matches allowed_domain (exact or wildcard subdomain)."""
    origin = origin_host.strip().lower()
    allowed = allowed_domain.strip().lower()
    if not origin or not allowed:
        return False
    if allowed == "*":
        return True
    if allowed.startswith("*."):
        base = allowed[2:]
        return origin == base or origin.endswith("." + base)
    return origin == allowed


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename while preserving its extension."""
    raw = (filename or "").strip()
    if not raw:
        raise ValidationError("Filename cannot be empty.")
    base = raw.replace("\\", "/").split("/")[-1]
    cleaned = SAFE_FILENAME_CHARS.sub("_", base).strip("._")
    if not cleaned:
        cleaned = "document.txt"
    return cleaned[:120]


def build_storage_object_path(tenant_id: str, source_id: str, original_filename: str) -> str:
    """Build a collision-proof private storage path: <tenant-id>/<source-id>/<safe-filename>."""
    safe_name = sanitize_filename(original_filename)
    return f"{tenant_id.strip()}/{source_id.strip()}/{safe_name}"


def validate_upload_metadata(filename: str, content_type: str, size_bytes: int, max_mb: int = 10) -> str:
    """Validate uploaded knowledge file extension, MIME type, and size."""
    safe_name = sanitize_filename(filename)
    lower_name = safe_name.lower()
    ext = ""
    if "." in lower_name:
        ext = "." + lower_name.rsplit(".", 1)[-1]
    if ext not in ALLOWED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported document format '{ext or 'unknown'}'. Allowed formats: PDF, DOCX, TXT, Markdown (.md), CSV."
        )
    max_bytes = max_mb * 1024 * 1024
    if size_bytes <= 0:
        raise ValidationError("Uploaded file is empty.")
    if size_bytes > max_bytes:
        raise ValidationError(f"File size ({size_bytes} bytes) exceeds maximum limit of {max_mb} MB.")
    mime = (content_type or "application/octet-stream").split(";")[0].strip().lower()
    if mime and mime not in ALLOWED_MIME_TYPES and not mime.startswith("text/"):
        raise ValidationError(f"Unsupported MIME type '{mime}'.")
    return ext
