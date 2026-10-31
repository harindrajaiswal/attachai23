"""Request and response schemas for Public, Admin, and System APIs.

Uses Pydantic v2 BaseModel when installed and provides identical validation
methods in environments running with standard library only.
"""

from typing import Any, Dict, List, Optional
from app.core.exceptions import ValidationError
from app.domain.models import WidgetConfig
from app.utils.validators import normalize_slug, validate_email_address, validate_hex_color


class ChatRequestSchema:
    def __init__(
        self,
        message: str,
        sessionId: Optional[str] = None,
        conversationId: Optional[str] = None,
        messageId: Optional[str] = None,
    ) -> None:
        if not message or not str(message).strip():
            raise ValidationError("Field 'message' is required and cannot be empty.")
        self.message = str(message).strip()
        self.sessionId = str(sessionId).strip() if sessionId else None
        self.conversationId = str(conversationId).strip() if conversationId else None
        self.messageId = str(messageId).strip() if messageId else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatRequestSchema":
        if not isinstance(data, dict):
            raise ValidationError("Invalid JSON request body.")
        return cls(
            message=data.get("message", ""),
            sessionId=data.get("sessionId") or data.get("session_id"),
            conversationId=data.get("conversationId") or data.get("conversation_id"),
            messageId=data.get("messageId") or data.get("message_id"),
        )


class FeedbackSubmitSchema:
    def __init__(
        self,
        conversationId: str,
        messageId: str,
        feedbackType: str,
        reason: Optional[str] = None,
    ) -> None:
        conv_id = str(conversationId or "").strip()
        msg_id = str(messageId or "").strip()
        fb_type = str(feedbackType or "").strip().lower()
        if not conv_id:
            raise ValidationError("Field 'conversationId' is required.")
        if not msg_id:
            raise ValidationError("Field 'messageId' is required.")
        if fb_type not in ("like", "dislike"):
            raise ValidationError("Field 'feedbackType' must be 'like' or 'dislike'.")
        self.conversationId = conv_id
        self.messageId = msg_id
        self.feedbackType = fb_type
        self.reason = str(reason).strip()[:1000] if reason is not None and str(reason).strip() else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeedbackSubmitSchema":
        if not isinstance(data, dict):
            raise ValidationError("Invalid JSON request body.")
        return cls(
            conversationId=data.get("conversationId") or data.get("conversation_id") or "",
            messageId=data.get("messageId") or data.get("message_id") or "",
            feedbackType=data.get("feedbackType") or data.get("feedback_type") or "",
            reason=data.get("reason"),
        )


class LeadSubmitSchema:
    def __init__(
        self,
        name: str,
        email: str,
        phone: str = "",
        requirement: str = "",
        message: str = "",
        conversationId: Optional[str] = None,
        idempotencyKey: Optional[str] = None,
    ) -> None:
        cleaned_name = (name or "").strip()
        if len(cleaned_name) < 2:
            raise ValidationError("Field 'name' must be at least 2 characters.")
        self.name = cleaned_name
        self.email = validate_email_address(email)
        self.phone = str(phone or "").strip()
        self.requirement = str(requirement or "").strip()
        self.message = str(message or "").strip()
        self.conversationId = str(conversationId).strip() if conversationId else None
        self.idempotencyKey = str(idempotencyKey).strip() if idempotencyKey else None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LeadSubmitSchema":
        if not isinstance(data, dict):
            raise ValidationError("Invalid JSON request body.")
        return cls(
            name=data.get("name", ""),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            requirement=data.get("requirement", ""),
            message=data.get("message", ""),
            conversationId=data.get("conversationId") or data.get("conversation_id"),
            idempotencyKey=data.get("idempotencyKey") or data.get("idempotency_key"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "requirement": self.requirement,
            "message": self.message,
            "conversationId": self.conversationId,
            "idempotencyKey": self.idempotencyKey,
        }


class TenantCreateSchema:
    @classmethod
    def validate(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValidationError("Invalid tenant payload.")
        b_name = str(data.get("business_name") or data.get("businessName") or "").strip()
        if not b_name:
            raise ValidationError("business_name is required.")
        raw_slug = str(data.get("slug") or b_name).strip()
        slug = normalize_slug(raw_slug)
        w_cfg = data.get("widget_config") or data.get("widgetConfig") or {}
        if isinstance(w_cfg, dict) and w_cfg.get("primary_color"):
            w_cfg["primary_color"] = validate_hex_color(str(w_cfg["primary_color"]))
        return {
            **data,
            "business_name": b_name,
            "slug": slug,
            "company_name": str(data.get("company_name") or data.get("companyName") or b_name).strip(),
            "assistant_name": str(data.get("assistant_name") or data.get("assistantName") or "AI Assistant").strip(),
            "widget_config": WidgetConfig.from_dict(w_cfg).to_dict(),
        }
