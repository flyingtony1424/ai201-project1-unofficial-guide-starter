"""
Milestone 3 — Document ingestion and chunking.

Loads the raw dorm-life documents from documents/, cleans them, and splits each
one into overlapping character chunks following the strategy in planning.md:

    Chunk size: 1500 characters
    Overlap:    300 characters

The cleaned chunks (with source metadata) are written to chunks.json so the
embedding/retrieval stage (Milestone 4) can consume them directly.

Run:  python ingest.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

# --- Configuration (from planning.md → Chunking Strategy) ----------------------

DOCUMENTS_DIR = Path(__file__).parent / "documents"
OUTPUT_FILE = Path(__file__).parent / "chunks.json"

CHUNK_SIZE = 1500      # characters per chunk
CHUNK_OVERLAP = 300    # characters shared between consecutive chunks


@dataclass
class Chunk:
    """One chunk of text plus the metadata needed for retrieval/attribution."""

    id: str            # e.g. "rate_my_dorm-0"
    source: str        # file name the chunk came from
    chunk_index: int   # position of this chunk within its source document
    text: str
    char_count: int


# --- Stage 1: load ------------------------------------------------------------

def _decode(raw_bytes: bytes) -> str:
    """Decode bytes, auto-detecting UTF-8 vs Windows-1252.

    Some sources (e.g. the Alligator export) are Windows-1252 encoded: their
    curly quotes/apostrophes/em-dashes use bytes like 0x92/0x97 that are invalid
    as standalone UTF-8 and would otherwise become "�" glyphs. We try strict
    UTF-8 first (preserves genuine UTF-8 files, including emoji), and fall back
    to cp1252 only when strict UTF-8 decoding fails.
    """
    try:
        return raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return raw_bytes.decode("cp1252", errors="replace")


def load_documents(directory: Path) -> dict[str, str]:
    """Read every .txt file in `directory`, keyed by file name.

    Only .txt files are loaded so the extension-less duplicate of
    gen_dormm_life.txt ("gen_dormlif") is skipped automatically.
    """
    docs: dict[str, str] = {}
    for path in sorted(directory.glob("*.txt")):
        docs[path.name] = _decode(path.read_bytes())
    return docs


# --- Stage 2: clean -----------------------------------------------------------

# Lines that are pure site chrome / boilerplate add noise without information.
# Each pattern is matched against a fully-stripped single line, so prose that
# merely mentions e.g. "Gainesville" inside a sentence is never affected.
_BOILERPLATE_PATTERNS = [
    re.compile(r"^\s*\w[\w ]*\blogo\b\s*$", re.IGNORECASE),  # "Prked logo"
    re.compile(r"^\s*_{3,}\s*$"),                            # "________"
    re.compile(r"^\s*-{3,}\s*$"),                            # "--------"
    re.compile(r"^\s*View all reviews\s*$", re.IGNORECASE),

    # Alligator (residence.alligator.com) per-dorm footer block, repeated ~20x:
    #   2700 SW 13th St. / Gainesville, FL / 32601 / 352-376-4482 / editor@...
    re.compile(r"^2700 SW 13th St\.?$", re.IGNORECASE),
    re.compile(r"^Gainesville, FL$", re.IGNORECASE),
    re.compile(r"^32601$"),
    re.compile(r"^352-376-4482$"),
    re.compile(r"^editor@alligator\.org$", re.IGNORECASE),

    # Prked (prked.com) site footer / parking ad chrome:
    re.compile(r"^.*Prked connects people.*driveways\.?\"?$", re.IGNORECASE),
    re.compile(r"^Parking when you need it\.?$", re.IGNORECASE),
    re.compile(r"^Extra income when you don['’]t\.?$", re.IGNORECASE),
    re.compile(r"^Our (services|policies)$", re.IGNORECASE),
    re.compile(r"^Resources$", re.IGNORECASE),
    re.compile(r"^Copyright ©.*Prked.*$", re.IGNORECASE),
    re.compile(r"^[;:.]$"),                                  # stray punctuation lines
]


def clean_text(text: str) -> str:
    """Normalize unicode/whitespace and drop obvious boilerplate lines."""
    # Normalize unicode (curly quotes, em dashes, emoji-adjacent forms) to a
    # consistent NFKC form, then drop the U+FFFD replacement char.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("�", "")

    # Normalize newlines and non-breaking spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if any(pat.match(stripped) for pat in _BOILERPLATE_PATTERNS):
            continue
        # Collapse runs of internal whitespace within a line.
        cleaned_lines.append(re.sub(r"[ \t]+", " ", stripped))

    text = "\n".join(cleaned_lines)

    # Collapse 3+ blank lines down to a single blank line (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --- Stage 3: chunk -----------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into chunks of ~`chunk_size` chars with `overlap` overlap.

    Chunks prefer to end on a whitespace boundary (within the last 20% of the
    window) so words aren't cut mid-token. The next chunk starts `overlap`
    characters before the previous chunk's end to preserve context that would
    otherwise be split across a boundary.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        # If we're not at the end of the document, try to break on whitespace
        # so we don't slice through a word.
        if end < n:
            window_start = start + int(chunk_size * 0.8)
            break_at = text.rfind(" ", window_start, end)
            newline_at = text.rfind("\n", window_start, end)
            break_at = max(break_at, newline_at)
            if break_at > start:
                end = break_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        # Step forward, leaving `overlap` characters of context.
        start = end - overlap

    return chunks


# --- Orchestration ------------------------------------------------------------

def build_chunks(directory: Path = DOCUMENTS_DIR) -> list[Chunk]:
    documents = load_documents(directory)
    all_chunks: list[Chunk] = []

    for source, raw in documents.items():
        cleaned = clean_text(raw)
        stem = Path(source).stem
        for i, piece in enumerate(chunk_text(cleaned)):
            all_chunks.append(
                Chunk(
                    id=f"{stem}-{i}",
                    source=source,
                    chunk_index=i,
                    text=piece,
                    char_count=len(piece),
                )
            )
    return all_chunks


def main() -> None:
    chunks = build_chunks()

    OUTPUT_FILE.write_text(
        json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # --- Summary report -------------------------------------------------------
    per_source: dict[str, int] = {}
    for c in chunks:
        per_source[c.source] = per_source.get(c.source, 0) + 1

    sizes = [c.char_count for c in chunks]
    print(f"Loaded {len(per_source)} documents")
    print(f"Produced {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    if sizes:
        print(f"Chunk size chars  min={min(sizes)}  "
              f"max={max(sizes)}  avg={sum(sizes) // len(sizes)}")
    print("\nChunks per source:")
    for source in sorted(per_source):
        print(f"  {source:<28} {per_source[source]}")
    print(f"\nWrote {OUTPUT_FILE.name}")


if __name__ == "__main__":
    main()
