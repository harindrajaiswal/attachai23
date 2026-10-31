"""Domain data models for tenants, knowledge chunks, conversations, leads, and RAG."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.domain.enums import (
    KnowledgeStatus,
    LeadStatus,
    NotificationStatus,
    SupportLevel,
    ThemeMode,
    WidgetPosition,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class WidgetConfig:
    primary_color: str = "#2563eb"
    theme_mode: str = ThemeMode.LIGHT.value
    position: str = WidgetPosition.BOTTOM_RIGHT.value
    width: int = 380
    height: int = 560
    border_radius: int = 14
    logo_url: str = ""
    avatar_url: str = ""
    launcher_icon: str = "chat"
    input_placeholder: str = "Ask a question..."
    show_branding: bool = True
    suggested_questions: List[str] = field(
        default_factory=lambda: [
            "What services do you offer?",
            "What are your pricing and hours?",
            "How can I contact support?",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "WidgetConfig":
        if not data or not isinstance(data, dict):
            return cls()
        sq = data.get("suggested_questions")
        if isinstance(sq, str):
            sq = [s.strip() for s in sq.split("\n") if s.strip()]
        elif not isinstance(sq, list):
            sq = [
                "What services do you offer?",
                "What are your pricing and hours?",
                "How can I contact support?",
            ]
        theme = str(data.get("theme_mode", ThemeMode.LIGHT.value)).lower()
        if theme not in (ThemeMode.LIGHT.value, ThemeMode.DARK.value):
            theme = ThemeMode.LIGHT.value
        pos = str(data.get("position", WidgetPosition.BOTTOM_RIGHT.value)).lower()
        if pos not in (WidgetPosition.BOTTOM_RIGHT.value, WidgetPosition.BOTTOM_LEFT.value):
            pos = WidgetPosition.BOTTOM_RIGHT.value
        return cls(
            primary_color=str(data.get("primary_color") or "#2563eb"),
            theme_mode=theme,
            position=pos,
            width=max(280, min(600, int(data.get("width", 380)))),
            height=max(360, min(840, int(data.get("height", 560)))),
            border_radius=max(0, min(32, int(data.get("border_radius", 14)))),
            logo_url=str(data.get("logo_url") or ""),
            avatar_url=str(data.get("avatar_url") or ""),
            launcher_icon=str(data.get("launcher_icon") or "chat"),
            input_placeholder=str(data.get("input_placeholder") or "Ask a question..."),
            show_branding=bool(data.get("show_branding", True)),
            suggested_questions=[str(q).strip() for q in sq if str(q).strip()][:6],
        )


@dataclass
class Tenant:
    id: str
    business_name: str
    slug: str
    company_name: str
    assistant_name: str = "AI Assistant"
    enabled: bool = True
    system_prompt: str = (
        "You are a helpful, accurate customer support assistant. "
        "Answer strictly using the provided knowledge base."
    )
    fallback_answer: str = (
        "I do not have enough verified information in our knowledge base to answer that accurately. "
        "Please leave your contact details and our team will assist you."
    )
    welcome_message: str = "Hello! How can I help you today?"
    language: str = "en"
    model_override: Optional[str] = None
    temperature: float = 0.2
    top_k: int = 5
    min_similarity: float = 0.25
    lead_capture_enabled: bool = True
    lead_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "require_email": True,
            "require_phone": False,
            "custom_prompt": "Leave your contact details so our team can follow up.",
        }
    )
    notification_settings: Dict[str, Any] = field(
        default_factory=lambda: {
            "email_enabled": False,
            "notify_emails": [],
            "webhook_url": None,
        }
    )
    widget_config: WidgetConfig = field(default_factory=WidgetConfig)
    analytics_settings: Dict[str, Any] = field(
        default_factory=lambda: {"retention_days": 90, "track_latency": True}
    )
    domains: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["widget_config"] = self.widget_config.to_dict()
        return d


@dataclass
class KnowledgeFileRecord:
    id: str
    tenant_id: str
    original_filename: str
    safe_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    status: str = KnowledgeStatus.PROCESSING.value
    active_version: int = 1
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeChunk:
    id: str
    tenant_id: str
    source_id: str
    source_filename: str
    source_version: int
    chunk_index: int
    content: str
    embedding: List[float] = field(default_factory=list)
    similarity: float = 0.0
    lexical_score: float = 0.0
    combined_score: float = 0.0
    is_active: bool = True
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroundingResult:
    passed: bool
    support_level: str
    validated_answer: str
    reasons: List[str] = field(default_factory=list)
    unverified_numbers: List[str] = field(default_factory=list)
    coverage_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LeadRecord:
    id: str
    tenant_id: str
    idempotency_key: str
    name: str
    email: str
    phone: str = ""
    requirement: str = ""
    message: str = ""
    conversation_id: Optional[str] = None
    status: str = LeadStatus.NEW.value
    assignee: str = ""
    internal_notes: str = ""
    notification_status: str = NotificationStatus.PENDING.value
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
