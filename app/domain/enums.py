"""Domain enumerations for Orbit AI."""

from enum import Enum


class SupportLevel(str, Enum):
    """Evidence support assessment level for RAG answers."""

    SUPPORTED = "SUPPORTED"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


class KnowledgeStatus(str, Enum):
    """Ingestion lifecycle status for uploaded knowledge documents."""

    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class LeadStatus(str, Enum):
    """Customer lead CRM workflow status."""

    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    WON = "won"
    LOST = "lost"


class NotificationStatus(str, Enum):
    """Notification outbox delivery status."""

    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class ThemeMode(str, Enum):
    """Widget visual theme mode."""

    LIGHT = "light"
    DARK = "dark"


class WidgetPosition(str, Enum):
    """Widget placement on the host website."""

    BOTTOM_RIGHT = "bottom-right"
    BOTTOM_LEFT = "bottom-left"
