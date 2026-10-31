"""Embedding provider abstraction and implementations.

Default LocalEmbeddingProvider produces deterministic 128-dimensional unit vectors
using token hashing, bigrams, and character trigrams so semantic & lexical similarity
work out-of-the-box without a second external API key.

OpenAICompatibleEmbeddingProvider validates that every returned vector matches
the configured dimension without silent truncation or padding (Fixes Defect #8).
"""

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from app.core.exceptions import ConfigurationError, ProviderError

WORD_RE = re.compile(r"[a-z0-9]{2,}")

# Lightweight synonym normalization for deterministic local embeddings
SYNONYM_MAP = {
    "cost": "price",
    "pricing": "price",
    "rates": "price",
    "fee": "price",
    "fees": "price",
    "subscription": "membership",
    "plan": "membership",
    "plans": "membership",
    "hours": "schedule",
    "open": "schedule",
    "opening": "schedule",
    "closing": "schedule",
    "time": "schedule",
    "cancel": "cancellation",
    "refund": "cancellation",
    "support": "contact",
    "phone": "contact",
    "email": "contact",
    "reach": "contact",
    "gym": "fitness",
    "workout": "fitness",
    "classes": "training",
    "delivery": "shipping",
}


class BaseEmbeddingProvider(ABC):
    """Contract for text embedding providers."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions <= 0:
            raise ConfigurationError("Embedding dimensions must be positive.")
        self.dimensions = dimensions

    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate validated embeddings for a batch of texts."""

    async def embed_query(self, text: str) -> List[float]:
        """Generate a validated embedding for a single query string."""
        res = await self.embed_texts([text])
        return res[0]

    @abstractmethod
    async def check_health(self) -> Dict[str, Any]:
        """Check embedding provider health and dimension contract."""

    def validate_vectors(self, vectors: List[List[float]]) -> List[List[float]]:
        """Strictly verify every vector matches self.dimensions (never truncate or pad)."""
        for idx, vec in enumerate(vectors):
            if not isinstance(vec, list) or len(vec) != self.dimensions:
                actual = len(vec) if isinstance(vec, list) else "non-list"
                raise ProviderError(
                    message=(
                        f"Embedding dimension mismatch at index {idx}: "
                        f"expected {self.dimensions} dimensions, received {actual}."
                    ),
                    status_code=502,
                    provider="embeddings",
                )
        return vectors


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Deterministic 128-dimensional feature-hashed L2-normalized embedding provider."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions != 128:
            raise ConfigurationError(
                f"LocalEmbeddingProvider strictly requires 128 dimensions, got {dimensions}."
            )
        super().__init__(dimensions=128)

    def _compute_single(self, text: str) -> List[float]:
        vec = [0.0] * self.dimensions
        cleaned = (text or "").lower().strip()
        if not cleaned:
            vec[0] = 1.0
            return vec

        tokens = WORD_RE.findall(cleaned)
        canonical_tokens = [SYNONYM_MAP.get(t, t) for t in tokens]

        # 1. Unigram features (weighted 2.0)
        for tok in canonical_tokens:
            digest = hashlib.sha256(f"u:{tok}".encode("utf-8")).digest()
            idx1 = digest[0] % self.dimensions
            sign1 = 1.0 if (digest[1] & 1) == 0 else -1.0
            idx2 = digest[2] % self.dimensions
            sign2 = 1.0 if (digest[3] & 1) == 0 else -1.0
            vec[idx1] += 2.0 * sign1
            vec[idx2] += 1.0 * sign2

        # 2. Bigram features (weighted 1.5)
        for i in range(len(canonical_tokens) - 1):
            bg = f"b:{canonical_tokens[i]}_{canonical_tokens[i + 1]}"
            digest = hashlib.md5(bg.encode("utf-8")).digest()
            idx = digest[0] % self.dimensions
            sign = 1.0 if (digest[1] & 1) == 0 else -1.0
            vec[idx] += 1.5 * sign

        # 3. Character 3-grams for morphological resilience (weighted 0.35)
        compact = "".join(canonical_tokens)[:400]
        for i in range(len(compact) - 2):
            tg = compact[i : i + 3]
            h = (ord(tg[0]) * 313 + ord(tg[1]) * 37 + ord(tg[2])) % self.dimensions
            vec[h] += 0.35

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 1e-9:
            vec = [round(v / norm, 6) for v in vec]
        else:
            vec[0] = 1.0
        return vec

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = [self._compute_single(t) for t in texts]
        return self.validate_vectors(vectors)

    async def check_health(self) -> Dict[str, Any]:
        sample = await self.embed_query("health probe")
        return {
            "ok": len(sample) == self.dimensions,
            "provider": "local",
            "model": "orbit-local-128d",
            "dimensions": self.dimensions,
        }


class OpenAICompatibleEmbeddingProvider(BaseEmbeddingProvider):
    """Remote OpenAI-compatible `/embeddings` provider with strict dimension validation."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int = 128,
        timeout_ms: int = 15000,
        http_handler: Optional[Callable[[str, Dict[str, str], Dict[str, Any]], Tuple[int, Any]]] = None,
    ) -> None:
        super().__init__(dimensions=dimensions)
        if not base_url or not api_key or not model:
            raise ConfigurationError(
                "EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL are required."
            )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.model = model.strip()
        self.timeout_sec = max(1.0, timeout_ms / 1000.0)
        self._http_handler = http_handler

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "dimensions": self.dimensions,
        }

        if self._http_handler is not None:
            status, body = self._http_handler(url, headers, payload)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    status = resp.status
                    body = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise ProviderError(
                    message=f"Embeddings provider failed with HTTP {exc.code}.",
                    status_code=502,
                    provider="openai-embeddings",
                ) from exc
            except Exception as exc:
                raise ProviderError(
                    message="Embeddings provider request failed or timed out.",
                    status_code=502,
                    provider="openai-embeddings",
                ) from exc

        if status >= 400 or not isinstance(body, dict) or "data" not in body:
            raise ProviderError(
                message="Invalid response from embeddings provider.",
                status_code=502,
                provider="openai-embeddings",
            )

        items = body.get("data") or []
        if len(items) != len(texts):
            raise ProviderError(
                message=f"Embeddings batch count mismatch: sent {len(texts)}, received {len(items)}.",
                status_code=502,
                provider="openai-embeddings",
            )
        vectors = [item.get("embedding") for item in items]
        return self.validate_vectors(vectors)

    async def check_health(self) -> Dict[str, Any]:
        try:
            vec = await self.embed_query("health check")
            return {
                "ok": len(vec) == self.dimensions,
                "provider": "openai-compatible",
                "model": self.model,
                "dimensions": self.dimensions,
            }
        except Exception as exc:
            return {
                "ok": False,
                "provider": "openai-compatible",
                "model": self.model,
                "dimensions": self.dimensions,
                "error": str(exc),
            }
