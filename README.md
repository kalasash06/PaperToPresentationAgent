# Paper-to-Presentation & Viva Agent

An agentic academic assistant. Upload a research paper (PDF) and chat with a team of agents that produce a
**sectioned summary**, a **downloadable slide deck (.pptx)**, **concept explanations**, and **viva questions with
model answers**, all grounded in the paper through retrieval (RAG) over ChromaDB.

## Architecture

```
PDF upload -> section-aware chunking -> embeddings -> ChromaDB
                                                         |
User message -> Intent Router (LLM, few-shot) -----------+
        |-- summary -> overview retrieval -> Summarizer Agent (+ optional reflection)
        |-- slides  -> overview retrieval -> Concept-Extractor -> Slide-Generator -> .pptx
        |-- viva    -> overview retrieval -> Concept-Extractor -> Viva Agent
        |-- concept -> Retrieval Planner decides -> (retrieve) -> Concept-Extractor (explain)
        `-- chat    -> Retrieval Planner decides -> (retrieve or answer from memory) -> Paper Assistant
```

| Agent | Responsibility | File / chain |
|---|---|---|
| Summarizer Agent | Sectioned summary using RAG over the paper | `summary_chain` |
| Concept-Extractor Agent | Extracts contributions, methods, results; explains concepts | `extract_chain`, `explain_chain` |
| Slide-Generator Agent | Structures content into slides, exports .pptx | `slides_chain`, `pptx_builder.py` |
| Viva Agent | Likely examiner questions with model answers | `viva_chain` |

## How the 7 acceptance criteria are met

| # | Criterion | Where |
|---|---|---|
| 1 | LLM integration | `config.py`: Gemini or Groq (API) or Ollama (local), chosen in `.env` |
| 2 | Interactive GUI | `app.py`: Streamlit chat UI, PDF upload, buttons, settings, .pptx download |
| 3 | Prompt engineering | `prompts.py`: few-shot (router, viva), Chain of Thought (extractor, explainer), reflection (fact-checker). System and user prompts are separate and shown in the app sidebar under "Prompts used" |
| 4 | LangChain core | `ChatPromptTemplate` objects in `prompts.py`; LCEL chains `prompt \| llm \| parser` in `agents.py` |
| 5 | Agentic RAG | `PaperAgent.decide_retrieval`: the agent decides whether to retrieve and rewrites the query; retrieved excerpts ground the answer |
| 6 | Intent detection and routing | `PaperAgent.route` classifies summary / slides / concept / viva / chat and `PaperAgent.run` dispatches to the right chain (keyword fallback if the LLM fails) |
| 7 | Vector database | `rag.py`: ChromaDB with similarity search, section metadata filtering |

**Conversation memory:** `app.py` keeps the last turns and the last task in `st.session_state` and passes them into the router and chains, so follow-ups like "make it shorter" or "harder ones" work.

## Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your API key (free Gemini or Groq key)
streamlit run app.py
```

The first upload downloads a small local embedding model (about 80 MB), after which indexing is offline.

## Demo flow

1. Upload a text-based research paper PDF (not a scanned image).
2. Click **Summarize paper**, then **Make slides** (download the .pptx), then **Viva questions**.
3. Ask free-form things: "Explain the loss function used here", "Give me harder viva questions", "Thanks".
4. Check the caption under each answer: it shows the detected intent, whether retrieval happened, and which paper sections were used.

## Tests

```bash
python test_basic.py
```
Runs offline with a fake LLM and fake vector store to verify routing, chain wiring, JSON handling and .pptx export.

## Project structure

```
app.py            Streamlit GUI
agents.py         LCEL chains, intent router, agentic-RAG decision, orchestrator
prompts.py        All system and user prompts
rag.py            PDF parsing, section-aware chunking, ChromaDB indexing and retrieval
pptx_builder.py   Slide JSON -> .pptx
config.py         LLM provider selection
test_basic.py     Offline tests
```

## Limitations

- Scanned PDFs need OCR first. Section detection relies on standard headings (Abstract, Introduction, Methods...), and falls back to treating the paper as one section.
- Two-column layouts can produce slightly jumbled text depending on the PDF.
- Generated content is grounded in retrieved excerpts but should still be reviewed by the student.
