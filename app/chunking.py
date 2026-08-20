import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    token_count: int


def chunk_text(text: str, size: int = 500, overlap: int = 50) -> list[TextChunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("overlap must be smaller than positive size")
    tokens = re.findall(r"\S+", text)
    if not tokens:
        return []
    step = size - overlap
    return [
        TextChunk(" ".join(tokens[start : start + size]), len(tokens[start : start + size]))
        for start in range(0, len(tokens), step)
    ]
