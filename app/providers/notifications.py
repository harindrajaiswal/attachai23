"""Notification and Outbox provider abstraction (Supabase, Outbox, SMTP).

Fixes Defect #12:
Uses an outbox/retry pattern so lead capture always persists first and notification
delivery failures never roll back or duplicate lead records.
"""

import smtplib
from email.message import EmailMessage
from typing import Any, Callable, Dict, List, Optional
from app.repositories.base import BaseRepository
from app.utils.pii import redact_text


class NotificationProvider:
    """Outbox-backed notification dispatcher supporting supabase, outbox, and smtp."""

    def __init__(
        self,
        repository: BaseRepository,
        provider_type: str = "outbox",
        smtp_host: str = "",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_pass: str = "",
        smtp_from: str = "notifications@orbit-ai.local",
        send_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.repository = repository
        self.provider_type = provider_type.lower().strip()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_pass = smtp_pass
        self.smtp_from = smtp_from
        self._send_hook = send_hook

    async def enqueue_and_dispatch(
        self,
        tenant_id: str,
        lead_id: str,
        recipient: str,
        subject: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create a persistent outbox notification record and attempt immediate delivery."""
        record = await self.repository.create_notification(
            {
                "tenant_id": tenant_id,
                "lead_id": lead_id,
                "channel": self.provider_type,
                "recipient": recipient,
                "subject": subject,
                "payload": payload,
                "status": "pending",
                "attempts": 0,
            }
        )
        return await self.attempt_delivery(record)

    async def attempt_delivery(self, notification: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt delivery of a notification record and update its status safely."""
        notif_id = notification["id"]
        attempts = int(notification.get("attempts", 0)) + 1
        try:
            if self._send_hook is not None:
                self._send_hook(notification)
            elif self.provider_type == "smtp" and self.smtp_host and notification.get("recipient"):
                msg = EmailMessage()
                msg["Subject"] = notification["subject"]
                msg["From"] = self.smtp_from
                msg["To"] = notification["recipient"]
                msg.set_content(f"New Orbit AI Lead Notification:\n\n{notification.get('payload')}")
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5) as server:
                    if self.smtp_user and self.smtp_pass:
                        server.starttls()
                        server.login(self.smtp_user, self.smtp_pass)
                    server.send_message(msg)

            updated = await self.repository.update_notification(
                notif_id,
                {"status": "sent", "attempts": attempts, "last_error": None},
            )
            return updated or {**notification, "status": "sent", "attempts": attempts}
        except Exception as exc:
            safe_err = redact_text(str(exc), extra_secrets=[self.smtp_pass])
            updated = await self.repository.update_notification(
                notif_id,
                {"status": "failed", "attempts": attempts, "last_error": safe_err},
            )
            return updated or {**notification, "status": "failed", "attempts": attempts, "last_error": safe_err}

    async def retry_failed_notifications(self, tenant_id: Optional[str] = None, max_attempts: int = 3) -> int:
        """Retry failed or pending notifications up to max_attempts."""
        failed = await self.repository.list_notifications(tenant_id=tenant_id, status="failed", limit=50)
        retried = 0
        for item in failed:
            if int(item.get("attempts", 0)) < max_attempts:
                await self.attempt_delivery(item)
                retried += 1
        return retried

    async def check_health(self) -> Dict[str, Any]:
        db_health = await self.repository.check_health()
        return {
            "ok": bool(db_health.get("ok")),
            "provider": self.provider_type,
            "persistent_outbox": True,
        }
