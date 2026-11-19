"""Word-based chunking that approximates tokens for retrieval."""

CHUNK_TOKENS = 800
CHUNK_OVERLAP = 100


def _split_words(text: str, size: int = CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []
    if size <= 0:
        raise ValueError("chunk size must be positive")
    step = max(size - overlap, 1)
    pieces: list[str] = []
    start = 0
    while start < len(words):
        piece = " ".join(words[start : start + size]).strip()
        if piece:
            pieces.append(piece)
        if start + size >= len(words):
            break
        start += step
    return pieces


def chunk_pages(
    pages: list[tuple[int | None, str]],
    size: int = CHUNK_TOKENS,
    overlap: int = CHUNK_OVERLAP,
) -> list[tuple[int, str, int | None]]:
    """Split page texts into overlapping chunks.

    Tokens are approximated as whitespace-separated words. Each result is
    ``(idx, text, page)`` with ``idx`` increasing across the whole file.
    """
    chunks: list[tuple[int, str, int | None]] = []
    idx = 0
    for page, text in pages:
        for piece in _split_words(text, size=size, overlap=overlap):
            chunks.append((idx, piece, page))
            idx += 1
    return chunks
