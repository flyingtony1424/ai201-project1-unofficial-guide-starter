# Project 1 Planning: The Unofficial Guide

> Write this document before you write any pipeline code.
> Your spec and architecture diagram are what you'll use to direct AI tools (Claude, Copilot, etc.) to generate your implementation — the more specific they are, the more useful the generated code will be.
> Update the Retrieval Approach and Chunking Strategy sections if you change your approach during implementation.
> Update this file before starting any stretch features.

---

## Domain

<!-- What domain did you choose? Why is this knowledge valuable and hard to find through official channels? -->

Domain: On campus dorm-life experience at UF. Most UF websites provide formal and outside information about dorms. But this domain, covers both the inside and outside information; thus providing students with a full picture when they ask about on campus housing.


---

## Documents

<!-- List your specific sources: URLs, subreddit names, forum threads, or file descriptions.
     Aim for at least 10 sources that together cover different subtopics or perspectives within your domain. -->

| # | Source | Description | URL or location |
|---|--------|-------------|-----------------|
| 1 | ratenydorm.com| dorm ratings at Univrsity of florida|	C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\rate_my_dorm.txt|
| 2 | residence.alligator.com| inside information of Dorms at Universit of florida|C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\dorm_data.txt |
| 3 | reddit|short comments | C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\reddit.txt|
| 4 |skill nation |dorm reviews on skill nation |C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\skillnation_guide.txt |
| 5 |dorm-dwellers.com |reviews on dorm dwellers |C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\gen_dormlif|
| 6 | prked.com|dorm reviews on prked |C:\Users\anony\Documents\GitHub\ai201-project1-unofficial-guide-starter\documents\best_dorm_review.txt|
| 7 | | | |
| 8 | | | |
| 9 | | | |
| 10 | | | |

---

## Chunking Strategy

<!-- How will you split documents into chunks?
     State your chunk size (in tokens or characters), overlap size, and explain why those
     numbers fit the structure of your documents.
     A review-heavy corpus warrants different chunking than a long FAQ. -->

**Chunk size: 1500 characters**

**Overlap: 300 characters**

**Reasoning: use of heavy data **

---

## Retrieval Approach

<!-- Which embedding model are you using (e.g., all-MiniLM-L6-v2 via sentence-transformers)?
     How many chunks will you retrieve per query (top-k)?
     If you were deploying this for real users and cost wasn't a constraint, what tradeoffs
     would you weigh in choosing a different embedding model — context length, multilingual
     support, accuracy on domain-specific text, latency? -->

**Embedding model:all-MiniLM-L6-v2 via sentence-transformers**

**Top-k:4**

**Production tradeoff reflection:accuracy on domain-specific text, multilingual support**

---

## Evaluation Plan

<!-- List your 5 test questions with their expected correct answers.
     Questions should be specific enough that you can judge whether the system's response
     is right or wrong. "What are good dining halls?" is too vague.
     "What do students say about wait times at [dining hall name] during lunch?" is testable. -->

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | what year was spring complex built?| 1995 |
| 2 | what acre land does lakeside residential complex occupy? |8.3 acre |
| 3 | How many apartments does lakeside have?| 135 |
| 4 | Where is Yulee Hall located on UF campus?|eastern side of UF's Campus |
| 5 | When was infinity hall originally opened?| 2015 |

---

## Anticipated Challenges

<!-- What could go wrong? Name at least two specific risks with reasoning.
     Consider: noisy or inconsistent documents, missing source attribution, off-topic
     retrieval, chunks that split key information across boundaries. -->

1. inconsistent documents

2. information contained in different chunks

---

## Architecture

<!-- Draw a diagram of your pipeline showing the five stages:
     Document Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation
     Label each stage with the tool or library you're using.
     You can use ASCII art, a Mermaid diagram, or embed a sketch as an image.
     You'll use this diagram as context when prompting AI tools to implement each stage. -->

Document Ingestion → Chunking → ChromaDB + all-MiniLM-L6-v2 → llama-3.3-70b-versatile → generation
---


## AI Tool Plan

<!-- For each part of the pipeline below, describe:
     - Which AI tool you plan to use (Claude, Copilot, ChatGPT, etc.)
     - What you'll give it as input (which sections of this planning.md, which requirements)
     - What you expect it to produce
     - How you'll verify the output matches your spec

     "I'll use AI to help me code" is not a plan.
     "I'll give Claude my Chunking Strategy section and ask it to implement chunk_text()
     with my specified chunk size and overlap" is a plan. -->

**Milestone 3 — Ingestion and chunking:**
I'll prompt Claude with my chunking strategy section and ask it to implement the chunk_text() function

**Milestone 4 — Embedding and retrieval:**
I'll use all-MiniLM-L6-v2 for embedding and ChromaDB for storage

**Milestone 5 — Generation and interface:**