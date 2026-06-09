# The Unofficial Guide — Project 1

> **How to use this template:**
> Complete each section *after* you've built and tested the corresponding part of your system.
> Do not write placeholder text — if a section isn't done yet, leave it blank and come back.
> Every section below is required for submission. One-liners will not receive full credit.

---

## Domain

This system answers questions about **on-campus dorm life at the University of Florida** — what it is actually like to live in each residence hall, drawn from student reviews and unofficial guides rather than the housing office's marketing pages.

Official UF Housing pages list amenities, square footage, and rates, but they don't tell you that Buckman Hall has no elevator and a third-floor walk-up, that Jennings is loud and social while Tolbert can feel isolating if you aren't in ROTC, or that there's a friendly campus cat named Boots in the Tolbert area. That lived-experience knowledge is scattered across Reddit threads, review sites, and blog posts. Pulling it into one grounded question-answering system gives an incoming student the "inside and outside" picture in one place when deciding where to live.

---

## Document Sources

I collected six text documents from six distinct sources covering ratings, journalism, peer reviews, and guides. (A seventh file, `gen_dormlif`, was a byte-identical duplicate of `gen_dormm_life.txt` and is excluded by loading only `.txt` files, so it is not double-counted.)

| # | Source | Type | URL or file path |
|---|--------|------|-----------------|
| 1 | RateMyDorm | Aggregated student dorm ratings/rankings | `documents/rate_my_dorm.txt` (ratemydorm.com) |
| 2 | The Independent Florida Alligator | Student-newspaper dorm profiles + quotes | `documents/dorm_data.txt` (residence.alligator.com) |
| 3 | Reddit | First-person dorm reviews / comments | `documents/reddit.txt` |
| 4 | Skill Nation | Incoming-freshman dorm guide | `documents/skillnation_guide.txt` |
| 5 | dorm-dwellers.com | General "dorm life at UF" blog post | `documents/gen_dormm_life.txt` |
| 6 | Prked | "Best dorms at UF" insider guide | `documents/best_dorm_review.txt` (prked.com) |

---

## Chunking Strategy

Implemented in [ingest.py](ingest.py) (`load_documents` → `clean_text` → `chunk_text`).

**Chunk size:** 1500 characters

**Overlap:** 300 characters

**Preprocessing before chunking:**
- **Encoding detection** — sources were a mix of UTF-8 and Windows-1252. Reading everything as UTF-8 turned curly quotes/apostrophes into `�` glyphs. `clean_text` decodes strict UTF-8 first and falls back to cp1252, then NFKC-normalizes, so `'`, `"`, `—`, and emoji all survive intact (0 replacement chars across the corpus).
- **Boilerplate removal** — line-level filters strip pure site chrome: the Alligator per-dorm footer (`2700 SW 13th St. / Gainesville, FL / 32601 / 352-376-4482 / editor@alligator.org`, repeated ~20×), the Prked site footer / parking ad, `"... logo"` lines, horizontal rules, and stray punctuation lines. Patterns match full standalone lines only, so prose that merely mentions "Gainesville" is untouched.
- **Whitespace normalization** — CRLF→LF, non-breaking spaces→spaces, collapsed internal whitespace, and 3+ blank lines reduced to a single paragraph break.

**Why these choices fit my documents:** This is a review- and profile-heavy corpus. A single dorm write-up (an Alligator profile or a long Reddit review) typically runs 1–2k characters and bundles the facts that matter together — name, location, build year, room style, pros/cons. 1500 characters keeps a full profile (or most of one) inside a single chunk so a question like "How many apartments does Lakeside have?" lands in one place. The 300-character (20%) overlap is deliberately generous because facts here sit right next to each other in dense prose; the overlap is meant to keep a fact and its subject together when a boundary falls between them. `chunk_text` also prefers to break on whitespace/newlines in the last 20% of the window so words aren't sliced. (As the Failure Case below shows, this still isn't always enough.)

**Final chunk count:** **90 chunks** across 6 documents (avg 1445 chars; min 325, max 1499). Per source: `dorm_data.txt` 54, `rate_my_dorm.txt` 10, `skillnation_guide.txt` 9, `best_dorm_review.txt` 8, `reddit.txt` 6, `gen_dormm_life.txt` 3.

---

## Embedding Model

Implemented in [retrieval.py](retrieval.py).

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, stored in a persistent **ChromaDB** collection configured for **cosine** similarity (embeddings are L2-normalized). Retrieval returns the **top-k = 4** chunks per query.

I chose MiniLM because it is fast, runs locally with no API cost, and its 384-dimension embeddings are more than adequate for short English review text. Its 256-token window is not a limitation here because my chunks are sized to be retrieved as whole units anyway.

**Production tradeoff reflection:** If I were deploying this for real users and cost weren't a constraint, the main axis I'd weigh is **accuracy on domain-specific text**. Dorm names ("Yulee," "Beaty," "Hume"), UF-specific jargon (Reitz Union, Gator Corner, Murphree Area, LLCs), and slangy review phrasing are exactly the kind of out-of-distribution tokens a small general model embeds weakly — my own "How many reviews did RateMyDorm collect?" miss (top similarity only 0.28) is a symptom of this. A larger hosted model (e.g. OpenAI `text-embedding-3-large` or Voyage) would likely separate these better. I'd also weigh **context length** (a bigger window would let me embed whole dorm profiles without splitting), **latency** (a local model keeps the UI instant; an API adds a round-trip per query), and **multilingual support** (not needed for this English-only UF corpus today, but relevant if I expanded to international-student forums).

---

## Grounded Generation

Implemented in [generate.py](generate.py) using **Groq's `llama-3.3-70b-versatile`** (OpenAI-compatible, free tier), initialized with `from groq import Groq` and `GROQ_API_KEY` loaded from `.env`. Temperature is set to **0** for deterministic, fact-focused answers.

**System prompt grounding instruction (verbatim):**

> "You are a helpful guide to dorm life at the University of Florida. Answer the question using ONLY the information in the provided documents. Do not use any outside knowledge or make assumptions beyond the documents. If the documents don't contain enough information to answer, say exactly: \"I don't have enough information on that.\" When you do answer, cite which document number(s) you used, e.g. [Doc 1]."

**Structural grounding choices (beyond the prompt):**
- Retrieved chunks are passed as **numbered, source-labeled blocks** — `[Doc 1] (source: dorm_data.txt)\n<text>` — so the model can attribute claims to a specific document number.
- A **low-relevance filter** drops any retrieved chunk below 0.15 cosine similarity *before* generation. If nothing clears the bar, the system returns the refusal string without even calling the LLM — this is what makes off-topic questions ("What is the GDP of France?") refuse cleanly instead of hallucinating.
- The refusal phrasing is enforced as an exact string so it's unambiguous in evaluation.

**How source attribution is surfaced in the response:** Two ways, by design. (1) The model cites **inline** with `[Doc N]` markers in its prose. (2) After generation, the code **programmatically appends** a deduplicated `Sources:` list of the originating document file names (e.g. `Sources: dorm_data.txt, skillnation_guide.txt`). In the Gradio UI ([app.py](app.py)) these sources render in a separate "Retrieved from" box. *(Known limitation: the appended list currently reflects all retrieved chunks fed as context, not strictly the subset the model cited.)*

---

## Evaluation Report

All five planning.md test questions were run end-to-end through the deployed pipeline (`retrieve` → `llama-3.3-70b-versatile`).

| # | Question | Expected answer | System response (summarized) | Retrieval quality | Response accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What year was Springs Complex built? | 1995 | "Springs Complex was built in 1995. [Doc 1]" | Relevant (top sim 0.52, `dorm_data-49`) | Accurate |
| 2 | What acreage does Lakeside Residential Complex occupy? | 8.3 acre | "...built on an 8.3-acre site [Doc 1]" | Relevant (top sim 0.70, `dorm_data-51`) | Accurate |
| 3 | How many apartments does Lakeside have? | 135 | "Lakeside Residential Complex has 135 apartments [Doc 1]" | Relevant (top sim 0.72, `dorm_data-51`) | Accurate |
| 4 | Where is Yulee Hall located on UF campus? | eastern side of UF's campus | "...on the eastern side of UF's campus, near the College of Education, the Arts, and Design, Construction & Planning [Doc 1–3]" | Relevant (top sim 0.73, `skillnation_guide-6`) | Accurate |
| 5 | When was Infinity Hall originally opened? | 2015 | "Infinity Hall was originally opened in 2015. [Doc 1, Doc 2]" | Relevant (top sim 0.68, `dorm_data-17`) | Accurate |

**Retrieval quality:** Relevant (the answer-bearing chunk ranked in the top-k for all 5 questions)
**Response accuracy:** Accurate (5/5)

All five core questions passed, so to test the system honestly I ran ~10 additional in-corpus probes. Most also succeeded (Beaty Towers height "over 160 feet," Buckman opened 1906, capacity "under 140"), but two failed — one is analyzed below.

---

## Failure Case Analysis

**Question that failed:** "What was Fletcher Hall originally named?" (Answer present in the corpus: **North Hall**.)

**What the system returned:** *"Fletcher Hall was not originally named Fletcher Hall, but the document does not explicitly state what it was originally named. However, it does mention that another hall, North Hall, was built in the late 1930s... but it does not confirm if Fletcher Hall was originally named North Hall [Doc 1]."* — i.e. it had the words in front of it but refused to make the connection.

**Root cause (tied to a specific pipeline stage):** This is a **chunking + retrieval boundary failure**, not a generation failure. The source sentence reads:

> "**Fletcher Hall is a residence hall** located in UF's historic district. Hogwarts-style arches and Collegiate Gothic-style architecture make up the Murphree area. **The hall**, built in the late 1930s, was **initially named North Hall**."

The subject ("Fletcher Hall is a residence hall...") and the fact ("**The hall** ... was initially named North Hall") are bound only by the pronoun-like reference *"The hall."* The chunk boundary fell **between** them: the binding subject sentence lives in chunk `dorm_data-44`, which retrieval did **not** return, while the top-ranked retrieved chunk `dorm_data-45` begins mid-sentence — `"...chitecture make up the Murphree area. The hall... was initially named North Hall. Fletcher Hall is located at the northeast end..."` In the retrieved context, *"The hall"* has no antecedent, and the very next sentence reintroduces "Fletcher Hall" as if it were a different subject. This is **compounded** by the fact that **"North Hall" is itself a separate, currently-existing dorm** in this corpus (it has its own profile in `dorm_data-5/6/7`), so the model reasonably reads "North Hall" as a different building and declines to assert the link.

So the 300-character overlap, which is meant to prevent exactly this, was defeated: the coreference chain spanned more than the overlap window and the answer-bearing chunk that *did* contain the full sentence was ranked just outside what generation could disambiguate.

**What I would change to fix it:** Two complementary changes. (1) **Chunk on semantic/sentence boundaries** (or prepend each dorm's name/heading to every chunk derived from its profile) so a fact is never separated from the dorm it describes — this directly removes the dangling-pronoun problem. (2) **Increase top-k or add a small re-ranking step** so the adjacent chunk holding the full subject sentence is also present in context. A lighter-weight mitigation would be to add the dorm name as chunk metadata and inject it into the prompt header for each `[Doc N]`.

---

## Spec Reflection

**One way the spec helped you during implementation:**
The planning.md Chunking Strategy and Retrieval Approach sections gave me concrete, non-negotiable parameters to code against instead of guessing. "1500 characters / 300 overlap" went straight into `chunk_text()`'s defaults, and "all-MiniLM-L6-v2 via sentence-transformers" plus the architecture diagram (`Chunking → ChromaDB + all-MiniLM-L6-v2 → llama-3.3-70b-versatile`) fixed both the embedding model and the exact tool at each stage. Because the spec named the pipeline order and the libraries up front, I could build each milestone as a standalone module (`ingest.py`, `retrieval.py`, `generate.py`, `app.py`) that plugged into the next without rework, and I could write the grounding prompt knowing it was the last stage feeding a Groq model.

**One way your implementation diverged from the spec, and why:**
planning.md specified **Top-k = 2**, but the implementation uses **Top-k = 4**. I changed this after seeing how the corpus behaves: with 1500-character chunks, a fact and its surrounding context sometimes straddle a chunk boundary (exactly the Fletcher Hall failure above), so retrieving only 2 chunks risks dropping the one that holds the answer. Widening to k=4 improves recall, and the grounding system prompt plus the 0.15 low-relevance filter keep precision high enough that the extra chunks don't pull the model off-topic. I also added two things the spec didn't mention but the real documents demanded — Windows-1252 encoding detection and repeated-footer boilerplate stripping — because without them the chunks were full of `�` glyphs and duplicated Alligator/Prked site chrome that diluted the embeddings. (planning.md should be updated to reflect k=4 to stay consistent with the build.)

---

## AI Usage

**Instance 1 — Ingestion & chunking (`ingest.py`)**

- *What I gave the AI:* My planning.md Chunking Strategy section (1500 chars / 300 overlap) and a request to implement a script that loads the documents, cleans them, and produces chunks at that size/overlap.
- *What it produced:* A three-stage script — `load_documents`, `clean_text`, `chunk_text` — where the chunker does fixed-size character windows with overlap but prefers to break on whitespace in the last 20% of each window, writing chunks (with source metadata) to `chunks.json`.
- *What I changed or overrode:* After printing random chunks I caught two real-data problems and directed fixes: (a) curly quotes were showing as `�` because some files are Windows-1252, so I had it add strict-UTF-8-then-cp1252 decoding; and (b) the Alligator and Prked site footers were repeating in nearly every chunk, so I had it add full-line boilerplate filters. These dropped the count from 91 to a cleaner 90 chunks.

**Instance 2 — Embedding/retrieval & generation (`retrieval.py`, `generate.py`)**

- *What I gave the AI:* My Retrieval Approach section and the architecture diagram, asking it to embed `chunks.json` with all-MiniLM-L6-v2, store in ChromaDB with source metadata, and write a `retrieve(query, top_k)` function; then to wire that into Groq's `llama-3.3-70b-versatile` with a grounding prompt and source attribution.
- *What it produced:* A ChromaDB + MiniLM index builder with a cosine-similarity `retrieve()` returning chunks plus similarity scores, and a `generate.answer()` that formats numbered source-labeled context, calls Groq, and appends a `Sources:` list.
- *What I changed or overrode:* I overrode **Top-k from the spec's 2 to 4** (see Spec Reflection) for better recall on boundary-split facts. I also directed the addition of the **0.15 low-relevance filter** so off-topic queries refuse before any LLM call, and confirmed the refusal string matched the exact wording I wanted for evaluation.
