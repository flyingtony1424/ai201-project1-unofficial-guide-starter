"""
Milestone 5 — Grounded generation.

Pipeline (per planning.md → Architecture diagram):

    retrieve(query) → llama-3.3-70b-versatile (Groq) → grounded answer + sources

This module ties the retrieval stage to the LLM:
  1. Retrieve the top-k chunks for a question (retrieval.retrieve).
  2. Build a prompt that passes those chunks as numbered context and instructs
     the model to answer ONLY from that context.
  3. Call Groq's llama-3.3-70b-versatile (OpenAI-compatible).
  4. Return the answer with source attribution appended programmatically.

Setup:
  - Copy .env.example to .env and set GROQ_API_KEY (free at https://console.groq.com).

Run:  python generate.py "When was Infinity Hall originally opened?"
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from retrieval import DEFAULT_TOP_K, retrieve

# --- Configuration ------------------------------------------------------------

LLM_MODEL = "llama-3.3-70b-versatile"
LOW_RELEVANCE_THRESHOLD = 0.15  # drop chunks below this cosine similarity
NO_ANSWER = "I don't have enough information on that."

SYSTEM_PROMPT = (
    "You are a helpful guide to dorm life at the University of Florida. "
    "Answer the question using ONLY the information in the provided documents. "
    "Do not use any outside knowledge or make assumptions beyond the documents. "
    f"If the documents don't contain enough information to answer, say exactly: "
    f"\"{NO_ANSWER}\" "
    "When you do answer, cite which document number(s) you used, e.g. [Doc 1]."
)

USER_PROMPT_TEMPLATE = """Here are the documents you may use to answer the question.

{context}

Question: {question}

Answer using only the documents above. Cite the document number(s) you used."""


# --- LLM client ---------------------------------------------------------------

def get_groq_client() -> Groq:
    """Initialize the Groq client from GROQ_API_KEY in .env."""
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "key from https://console.groq.com"
        )
    return Groq(api_key=api_key)


# --- Prompt construction ------------------------------------------------------

def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered, source-labeled context blocks."""
    blocks = []
    for i, ch in enumerate(chunks, start=1):
        blocks.append(
            f"[Doc {i}] (source: {ch['source']})\n{ch['text']}"
        )
    return "\n\n".join(blocks)


# --- Generation ---------------------------------------------------------------

def answer(question: str, top_k: int = DEFAULT_TOP_K) -> dict:
    """Answer a question with grounded generation + source attribution.

    Returns:
        {
            "question": str,
            "answer":   str,   # model answer with [Doc N] citations
            "sources":  list[str],  # unique source file names used as context
            "chunks":   list[dict], # the retrieved chunks (after filtering)
        }
    """
    retrieved = retrieve(question, top_k=top_k)

    # Filter out clearly off-topic chunks so weak retrieval doesn't drag the
    # model toward hallucinating from irrelevant context.
    chunks = [c for c in retrieved if c["similarity"] >= LOW_RELEVANCE_THRESHOLD]

    if not chunks:
        return {
            "question": question,
            "answer": NO_ANSWER,
            "sources": [],
            "chunks": [],
        }

    context = format_context(chunks)
    client = get_groq_client()

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,  # deterministic, fact-focused answers
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    context=context, question=question
                ),
            },
        ],
    )
    response_text = completion.choices[0].message.content.strip()

    # Programmatic source attribution: list unique sources, order preserved.
    sources: list[str] = []
    for c in chunks:
        if c["source"] not in sources:
            sources.append(c["source"])

    return {
        "question": question,
        "answer": response_text,
        "sources": sources,
        "chunks": chunks,
    }


def format_response(result: dict) -> str:
    """Human-readable answer block with appended source attribution."""
    lines = [result["answer"]]
    if result["sources"]:
        lines.append("")
        lines.append("Sources: " + ", ".join(result["sources"]))
    return "\n".join(lines)


# --- CLI ----------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "When was Infinity Hall originally opened?"

    print(f"Q: {question}\n")
    result = answer(question)
    print(format_response(result))


if __name__ == "__main__":
    main()
