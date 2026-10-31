"""Configurable paragraph-aware document chunking utility.

Defaults:
- chunk_size: 900 characters
- chunk_overlap: 120 characters
Preserves paragraph and sentence boundaries whenever possible.
"""

import re
from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> List[str]:
    """Split extracted document text into overlapping chunks preserving paragraphs."""
    if not text or not text.strip():
        return []

    if chunk_size < 100:
        chunk_size = 100
    if chunk_overlap < 0:
        chunk_overlap = 0
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 4

    normalized = re.sub(r"\r\n?", "\n", text).strip()
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", normalized) if p.strip()]
    if not paragraphs:
        return []

    chunks: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    def flush_current() -> None:
        nonlocal current_parts, current_len
        if not current_parts:
            return
        combined = "\n\n".join(current_parts).strip()
        if combined:
            chunks.append(combined)
        if chunk_overlap > 0 and combined:
            tail = combined[-chunk_overlap:].strip()
            # Snap overlap tail to word boundary
            if " " in tail and len(tail) > 25:
                tail = tail.split(" ", 1)[-1]
            current_parts = [tail] if tail else []
            current_len = len(tail)
        else:
            current_parts = []
            current_len = 0

    for para in paragraphs:
        if len(para) <= chunk_size:
            added_len = len(para) + (2 if current_parts else 0)
            if current_len + added_len <= chunk_size:
                current_parts.append(para)
                current_len += added_len
            else:
                flush_current()
                if current_len + len(para) + 2 <= chunk_size:
                    current_parts.append(para)
                    current_len += len(para) + 2
                else:
                    current_parts = [para]
                    current_len = len(para)
        else:
            # Paragraph itself exceeds chunk_size; split by sentence/window
            if current_parts:
                flush_current()
                current_parts = []
                current_len = 0

            sentences = re.split(r"(?<=[.!?])\s+", para)
            buf = ""
            for sent in sentences:
                if len(sent) > chunk_size:
                    if buf:
                        chunks.append(buf.strip())
                        buf = ""
                    step = max(chunk_size - chunk_overlap, 50)
                    for idx in range(0, len(sent), step):
                        sub = sent[idx : idx + chunk_size].strip()
                        if sub:
                            chunks.append(sub)
                elif len(buf) + len(sent) + 1 <= chunk_size:
                    buf = f"{buf} {sent}".strip() if buf else sent
                else:
                    if buf:
                        chunks.append(buf.strip())
                    overlap_prefix = buf[-chunk_overlap:].strip() if (buf and chunk_overlap > 0) else ""
                    if overlap_prefix and len(overlap_prefix) + len(sent) + 1 <= chunk_size:
                        buf = f"{overlap_prefix} {sent}".strip()
                    else:
                        buf = sent
            if buf:
                current_parts = [buf]
                current_len = len(buf)

    if current_parts:
        final_chunk = "\n\n".join(current_parts).strip()
        if final_chunk and (not chunks or final_chunk != chunks[-1]):
            chunks.append(final_chunk)

    return chunks
