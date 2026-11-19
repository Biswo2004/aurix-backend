"""BM25 retrieval and compact tabular summaries for uploaded files."""

import io
import logging
import re
import uuid
from dataclasses import dataclass

import pandas as pd
from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Chunk, File

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)
_SHEET_START = re.compile(r"^--- sheet: (.+) ---$", re.MULTILINE)
_TABULAR_MIMES = {
    "text/csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
_SUMMARY_ROW_CAP = 5000
_SUMMARY_CHAR_CAP = 4000


@dataclass(frozen=True)
class ScoredChunk:
    file_id: uuid.UUID
    file_name: str
    page: int | None
    text: str
    score: float


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN.finditer(text)]


def rank_chunks(rows: list[ScoredChunk], query: str, limit: int = 5) -> list[ScoredChunk]:
    """Return the highest-scoring chunks for ``query`` using BM25."""
    usable = [row for row in rows if row.text.strip()]
    if not usable or limit <= 0:
        return []
    query_tokens = _tokens(query)
    if not query_tokens:
        return usable[:limit]
    corpus = [_tokens(row.text) or ["_"] for row in usable]
    try:
        scores = BM25Okapi(corpus).get_scores(query_tokens)
    except Exception as exc:
        logger.info("bm25 failed: %s", exc.__class__.__name__)
        return usable[:limit]
    ranked = sorted(zip(usable, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    return [
        ScoredChunk(
            file_id=row.file_id,
            file_name=row.file_name,
            page=row.page,
            text=row.text,
            score=float(score),
        )
        for row, score in ranked[:limit]
    ]


def _frame_summary(frame: pd.DataFrame, label: str) -> str:
    working = frame if len(frame) <= _SUMMARY_ROW_CAP else frame.head(_SUMMARY_ROW_CAP)
    note = "" if len(frame) <= _SUMMARY_ROW_CAP else f" (summary uses the first {_SUMMARY_ROW_CAP} rows)"
    columns = ", ".join(str(column) for column in frame.columns)
    dtypes = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    head = frame.head(5).to_csv(index=False)
    try:
        described = working.describe(include="all").to_string()
    except Exception:
        described = "unavailable"
    text = (
        f"{label}{note}\n"
        f"columns: {columns}\n"
        f"shape: {frame.shape[0]} rows x {frame.shape[1]} columns\n"
        f"dtypes: {dtypes}\n"
        f"head:\n{head}\n"
        f"describe:\n{described}"
    )
    return text[:_SUMMARY_CHAR_CAP]


def _summarize_csv(name: str, text: str) -> str:
    frame = pd.read_csv(io.StringIO(text))
    return _frame_summary(frame, f"Tabular summary for {name}")


def _summarize_xlsx(name: str, text: str) -> str:
    matches = list(_SHEET_START.finditer(text))
    if not matches:
        frame = pd.read_csv(io.StringIO(text))
        return _frame_summary(frame, f"Tabular summary for {name}")
    sections: list[str] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end]
        end_marker = body.find("--- end sheet ---")
        if end_marker >= 0:
            body = body[:end_marker]
        sheet = match.group(1).strip()
        frame = pd.read_csv(io.StringIO(body.strip()))
        sections.append(_frame_summary(frame, f"Tabular summary for {name} / sheet {sheet}"))
    return "\n\n".join(sections)[:_SUMMARY_CHAR_CAP]


def summarize_tabular_file(name: str, mime: str, extracted_text: str) -> str | None:
    """Build a compact pandas summary for CSV and XLSX text, if it parses."""
    if mime not in _TABULAR_MIMES or not extracted_text.strip():
        return None
    try:
        if mime == "text/csv":
            return _summarize_csv(name, extracted_text)
        return _summarize_xlsx(name, extracted_text)
    except Exception as exc:
        logger.info("tabular summary skipped file=%s error=%s", name, exc.__class__.__name__)
        return None


async def load_conversation_chunks(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: str,
) -> list[ScoredChunk]:
    result = await session.execute(
        select(Chunk, File)
        .join(File, Chunk.file_id == File.id)
        .where(File.conversation_id == conversation_id, File.user_id == user_id)
        .order_by(File.created_at, Chunk.idx)
    )
    rows: list[ScoredChunk] = []
    for chunk, file in result.all():
        rows.append(
            ScoredChunk(
                file_id=file.id,
                file_name=file.name,
                page=chunk.page,
                text=chunk.text,
                score=0.0,
            )
        )
    return rows


async def load_tabular_summaries(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    user_id: str,
) -> list[tuple[uuid.UUID, str, str]]:
    result = await session.execute(
        select(File).where(File.conversation_id == conversation_id, File.user_id == user_id)
    )
    summaries: list[tuple[uuid.UUID, str, str]] = []
    for file in result.scalars().all():
        summary = summarize_tabular_file(file.name, file.mime, file.extracted_text)
        if summary:
            summaries.append((file.id, file.name, summary))
    return summaries
