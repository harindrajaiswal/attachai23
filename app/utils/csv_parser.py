"""RFC 4180 compliant CSV parser for knowledge ingestion and lead export.

Fixes Defect #14:
Properly handles quoted fields, escaped quotes (""), commas inside quotes,
and multiline fields inside quotes using Python's standard csv module.
"""

import csv
import io
from typing import Any, Dict, List


def parse_csv_to_structured_text(raw_bytes: bytes) -> str:
    """Parse CSV bytes into structured, readable row blocks for RAG chunking."""
    try:
        text = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw_bytes.decode("latin-1", errors="replace")

    text = text.strip()
    if not text:
        return ""

    reader = csv.reader(io.StringIO(text, newline=""), dialect="excel")
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if not rows:
        return ""

    headers = [h.strip() or f"Column_{i + 1}" for i, h in enumerate(rows[0])]
    if len(rows) == 1:
        return " | ".join(headers)

    blocks: List[str] = []
    for row_idx, row in enumerate(rows[1:], start=1):
        parts: List[str] = []
        for col_idx, val in enumerate(row):
            cleaned_val = val.strip()
            if not cleaned_val:
                continue
            header_name = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx + 1}"
            parts.append(f"{header_name}: {cleaned_val}")
        if parts:
            blocks.append(f"Record {row_idx} — " + "; ".join(parts))

    return "\n\n".join(blocks)


def generate_csv_string(headers: List[str], rows: List[Dict[str, Any]]) -> str:
    """Generate an RFC 4180 compliant CSV string from dictionaries."""
    output = io.StringIO(newline="")
    writer = csv.writer(output, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(headers)
    for row in rows:
        writer.writerow([row.get(h, "") if row.get(h, "") is not None else "" for h in headers])
    return output.getvalue()
