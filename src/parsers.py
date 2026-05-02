"""Input parsers for Streamlit demo upload modes."""

from __future__ import annotations

import re
from email import policy
from email.parser import BytesParser
from io import BytesIO
from typing import Tuple

import pandas as pd
from PyPDF2 import PdfReader

SPEAKER_RE = re.compile(r"^(Customer|Sales\s*Rep|Agent|Rep|Client|Prospect)\s*[:\-]\s*", re.IGNORECASE)


def parse_uploaded_file(uploaded_file):
    """Route uploaded file to the appropriate parser."""
    name = (getattr(uploaded_file, "name", "") or "").lower()
    if name.endswith(".txt"):
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        return parse_plain_text(content), None
    if name.endswith(".csv"):
        return parse_csv(uploaded_file)
    if name.endswith(".pdf"):
        return parse_pdf(uploaded_file), None
    if name.endswith(".eml"):
        return parse_email_file(uploaded_file), None

    content = uploaded_file.read().decode("utf-8", errors="ignore")
    return parse_plain_text(content), None


def parse_plain_text(text: str) -> str:
    """Normalize plain text and infer speaker labels if missing."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""

    has_labels = any(SPEAKER_RE.match(line) for line in lines)
    if has_labels:
        return "\n".join(lines)

    normalized = []
    for idx, line in enumerate(lines):
        speaker = "Customer" if idx % 2 == 0 else "Sales Rep"
        normalized.append(f"{speaker}: {line}")
    return "\n".join(normalized)


def parse_email_file(uploaded_file) -> str:
    """Parse .eml thread into a simple chronological dialogue."""
    raw = uploaded_file.read()
    msg = BytesParser(policy=policy.default).parsebytes(raw)

    subject = msg.get("Subject", "Unknown Subject")
    from_addr = msg.get("From", "Unknown Sender")
    to_addr = msg.get("To", "Unknown Recipient")
    sent_date = msg.get("Date", "Unknown Date")

    body = _extract_email_body(msg)
    parts = re.split(
        r"(?:\nOn .+ wrote:|--- Original Message ---|--- Forwarded message ---)",
        body,
        flags=re.IGNORECASE,
    )
    parts = [p.strip() for p in parts if p and p.strip()]
    parts.reverse()

    labeled = []
    for idx, part in enumerate(parts):
        speaker = "Sales Rep" if idx % 2 == 0 else "Customer"
        labeled.append(f"{speaker}: {part}")

    header = (
        f"[Email Thread]\nSubject: {subject}\nFrom: {from_addr}\n"
        f"To: {to_addr}\nDate: {sent_date}\n"
    )
    return header + "\n" + "\n".join(labeled)


def _extract_email_body(msg) -> str:
    """Extract readable text body from email message."""
    if msg.is_multipart():
        chunks = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                chunks.append(part.get_content())
        return "\n".join(chunks).strip()
    return str(msg.get_content()).strip()


def parse_csv(uploaded_file) -> Tuple[str, pd.DataFrame]:
    """Parse CSV and return first-row conversation plus full DataFrame."""
    df = pd.read_csv(uploaded_file)
    text_col = detect_text_column(df)
    first_text = parse_plain_text(str(df[text_col].iloc[0])) if len(df) > 0 else ""
    return first_text, df


def detect_text_column(df: pd.DataFrame) -> str:
    """Find most likely conversation text column in a dataframe."""
    preferred = ["full_text", "conversation", "text", "dialogue", "message", "body", "content", "transcript"]
    lower_map = {col.lower(): col for col in df.columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]

    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if not object_cols:
        raise ValueError("No text-like columns found in CSV.")

    lengths = []
    for col in object_cols:
        mean_len = df[col].fillna("").astype(str).str.len().mean()
        lengths.append((col, float(mean_len)))
    lengths.sort(key=lambda x: x[1], reverse=True)
    return lengths[0][0]


def parse_pdf(uploaded_file) -> str:
    """Extract text from PDF and normalize as conversation text."""
    reader = PdfReader(BytesIO(uploaded_file.read()))
    pages = [page.extract_text() or "" for page in reader.pages]
    return parse_plain_text("\n".join(pages))
