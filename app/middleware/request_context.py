"""Request ID and latency context tracking."""

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class RequestContext:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_monotonic: float = field(default_factory=time.monotonic)
    client_ip: str = "127.0.0.1"

    def elapsed_ms(self) -> int:
        return max(1, int((time.monotonic() - self.start_monotonic) * 1000))
