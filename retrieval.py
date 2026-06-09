"""
Milestone 4 — Embedding and retrieval.

Pipeline (per planning.md → Retrieval Approach & Architecture diagram):

    Chunking (chunks.json) → all-MiniLM-L6-v2 → ChromaDB → retrieve(query)

This module:
  1. Loads the cleaned chunks produced by ingest.py (chunks.json).
  2. Embeds them with all-MiniLM-L6-v2 (sentence-transformers).
  3. Stores them in a persistent ChromaDB collection, keeping source metadata.
  4. Exposes retrieve(query, top_k=4) -> the most relevant chunks + sources.

Embedding model: all-MiniLM-L6-v2  (planning.md: Retrieval Approach)
Top-k:           4                 (chunks returned per query)

Run `python retrieval.py` to (re)build the index and run a quick demo query.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# --- Configuration ------------------------------------------------------------

BASE_DIR = Path(__file__).parent
CHUNKS_FILE = BASE_DIR / "chunks.json"
CHROMA_DIR = BASE_DIR / "chroma_db"          # persistent vector store on disk

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "uf_dorm_guide"
DEFAULT_TOP_K = 4

# cosine similarity suits normalized sentence-transformer embeddings better than
# the default squared-L2 distance.
COLLECTION_METADATA = {"hnsw:space": "cosine"}


# --- Lazy singletons ----------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the all-MiniLM-L6-v2 model once and reuse it."""
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def get_client() -> chromadb.ClientAPI:
    """Persistent ChromaDB client so the index survives across runs."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts into embedding vectors."""
    vectors = get_embedder().encode(
        texts,
        show_progress_bar=False,
        normalize_embeddings=True,  # unit vectors -> cosine distance is clean
    )
    return vectors.tolist()


# --- Indexing -----------------------------------------------------------------

def load_chunks(path: Path = CHUNKS_FILE) -> list[dict]:
    """Load chunks emitted by the ingestion pipeline (ingest.py)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run `python ingest.py` first to produce it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_index(reset: bool = True) -> chromadb.Collection:
    """Embed all chunks and (re)store them in ChromaDB with source metadata.

    Each chunk is stored with:
      - id        : stable chunk id (e.g. "dorm_data-18")
      - document  : the chunk text
      - embedding : all-MiniLM-L6-v2 vector
      - metadata  : {source, chunk_index, char_count}
    """
    client = get_client()
    if reset:
        # Drop any previous build so re-running doesn't duplicate documents.
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
    )

    chunks = load_chunks()
    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "chunk_index": c["chunk_index"],
            "char_count": c["char_count"],
        }
        for c in chunks
    ]
    embeddings = embed_texts(documents)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return collection


def get_collection(build_if_empty: bool = True) -> chromadb.Collection:
    """Return the collection, building it from chunks.json if it's empty."""
    client = get_client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA,
    )
    if build_if_empty and collection.count() == 0:
        collection = build_index(reset=False)
    return collection


# --- Retrieval ----------------------------------------------------------------

def retrieve(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top-k most relevant chunks for `query`.

    Each result is a dict:
        {
            "id":         chunk id,
            "text":       chunk text,
            "source":     originating document file name,
            "similarity": cosine similarity in [0, 1] (higher = more relevant),
            "metadata":   full stored metadata,
        }
    Results are ordered most-relevant first.
    """
    collection = get_collection()
    query_embedding = embed_texts([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )

    hits: list[dict] = []
    # query() returns lists-of-lists (one inner list per query); we sent one.
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for cid, doc, meta, dist in zip(ids, documents, metadatas, distances):
        hits.append(
            {
                "id": cid,
                "text": doc,
                "source": meta.get("source", "unknown"),
                "similarity": round(1.0 - dist, 4),  # cosine distance -> similarity
                "metadata": meta,
            }
        )
    return hits


# --- CLI / demo ---------------------------------------------------------------

def main() -> None:
    print(f"Building index from {CHUNKS_FILE.name} "
          f"with {EMBED_MODEL_NAME} ...")
    collection = build_index(reset=True)
    print(f"Indexed {collection.count()} chunks into "
          f"ChromaDB collection '{COLLECTION_NAME}' at {CHROMA_DIR.name}/\n")

    demo_query = "When was Infinity Hall originally opened?"
    print(f"Demo query: {demo_query!r}  (top_k={DEFAULT_TOP_K})")
    print("=" * 70)
    for rank, hit in enumerate(retrieve(demo_query), start=1):
        preview = hit["text"].replace("\n", " ")[:200]
        print(f"#{rank}  [{hit['source']}]  id={hit['id']}  "
              f"similarity={hit['similarity']}")
        print(f"    {preview}...")
        print("-" * 70)


if __name__ == "__main__":
    main()
