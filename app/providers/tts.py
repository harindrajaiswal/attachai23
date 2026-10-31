"""Optional server-only Text-to-Speech (TTS) provider.

Section 11 & 16 requirement:
Recognizes TTS_API_KEY as an optional server-only configuration value.
Does NOT pretend TTS is enabled merely because TTS_API_KEY is present;
stays disabled unless provider, endpoint, and model/voice are also supplied.
Never exposes TTS_API_KEY to clients or logs.
"""

from typing import Any, Dict
from app.core.config import Settings


class TtsProvider:
    """Server-only optional TTS status and configuration adapter."""

    def __init__(self, settings: Settings) -> None:
        self._api_key = settings.tts_api_key
        self.provider = settings.tts_provider
        self.base_url = settings.tts_base_url
        self.model = settings.tts_model
        self.voice = settings.tts_voice
        self.enabled = settings.tts_enabled

    async def check_health(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "enabled": self.enabled,
            "provider": self.provider or "disabled",
            "key_configured": bool(self._api_key),
            "note": (
                "TTS active"
                if self.enabled
                else "TTS disabled (optional server-only capability; not required for chat or RAG)"
            ),
        }
