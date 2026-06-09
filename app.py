"""
Milestone 5 — Query interface (Gradio web UI).

A minimal web front-end over the full RAG pipeline:

    question → retrieve (ChromaDB + all-MiniLM-L6-v2) → llama-3.3-70b-versatile
             → grounded answer + source attribution

The end-to-end logic lives in generate.answer(); this file is just the UI.

Run:  python app.py     (then open the printed local URL)
"""

from __future__ import annotations

import gradio as gr

from generate import answer

EXAMPLE_QUESTIONS = [
    "When was Infinity Hall originally opened?",
    "How many apartments does Lakeside have?",
    "Where is Yulee Hall located on UF campus?",
    "Which dorms are best for honors students?",
]


def handle_query(question: str):
    """Run one question through the pipeline and format UI outputs."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", ""

    result = answer(question)
    sources = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="UF Dorm Guide — The Unofficial Guide") as demo:
    gr.Markdown(
        "# 🐊 The Unofficial UF Dorm Guide\n"
        "Ask about on-campus housing at the University of Florida. "
        "Answers are grounded **only** in the collected student reviews and "
        "dorm write-ups — if the documents don't cover it, the guide will say so."
    )

    inp = gr.Textbox(
        label="Your question",
        placeholder="e.g. When was Infinity Hall originally opened?",
    )
    btn = gr.Button("Ask", variant="primary")

    answer_box = gr.Textbox(label="Answer", lines=8)
    sources_box = gr.Textbox(label="Retrieved from", lines=4)

    gr.Examples(examples=EXAMPLE_QUESTIONS, inputs=inp)

    btn.click(handle_query, inputs=inp, outputs=[answer_box, sources_box])
    inp.submit(handle_query, inputs=inp, outputs=[answer_box, sources_box])


if __name__ == "__main__":
    demo.launch()
