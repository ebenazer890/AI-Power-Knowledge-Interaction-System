from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import pdfplumber


@dataclass(frozen=True)
class PdfPageChunk:
    page: int
    text: str


def extract_pdf_text(path: str, *, max_pages: Optional[int] = None) -> List[PdfPageChunk]:
    chunks: List[PdfPageChunk] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        limit = min(page_count, max_pages) if max_pages else page_count
        for i in range(limit):
            page = pdf.pages[i]
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                chunks.append(PdfPageChunk(page=i + 1, text=text))
    return chunks


def extract_pdf_tables(path: str, *, max_pages: Optional[int] = None) -> List[Tuple[int, List[List[Optional[str]]]]]:
    """Returns list of (page_number, tables) where tables is a list of rows."""
    out: List[Tuple[int, List[List[Optional[str]]]]] = []
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        limit = min(page_count, max_pages) if max_pages else page_count
        for i in range(limit):
            page = pdf.pages[i]
            tables = page.extract_tables() or []
            if tables:
                # Flatten: for each table we keep its rows
                for t in tables:
                    out.append((i + 1, t))
    return out


def chunk_text(text: str, *, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    text = " ".join(text.split())
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def page_chunks_to_passages(pages: Iterable[PdfPageChunk]) -> List[str]:
    passages: List[str] = []
    for p in pages:
        for c in chunk_text(p.text):
            passages.append(f"[page {p.page}] {c}")
    return passages
