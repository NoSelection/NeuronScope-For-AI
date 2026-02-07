"""Shared utilities for PDF report generation."""


def safe_text(text: str) -> str:
    """Sanitize text for fpdf2's built-in Helvetica font (latin-1 only).

    Replaces any character outside latin-1 range with '?' to prevent
    FPDFUnicodeEncodingException. Model tokens can contain arbitrary
    Unicode (emoji, replacement chars, CJK, etc.).
    """
    return text.encode("latin-1", errors="replace").decode("latin-1")
