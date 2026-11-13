"""The agent team and the intent-routing orchestrator.

Every agent is an LCEL chain:  ChatPromptTemplate | LLM | OutputParser
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

import prompts as P
from rag import PaperIndex, format_context, overview_docs, retrieve, sources_of

INTENTS = {"summary", "slides", "concept", "viva", "chat"}


@dataclass
class Result:
    intent: str                      # routed intent
    retrieved: bool                  # did the agent retrieve from ChromaDB?
    query: Optional[str]             # retrieval query used (if any)
    kind: str                        # "text" | "slides" | "viva"
    data: Any                        # str for text, list[dict] for slides / viva
    memo: str                        # short note stored in conversation memory
    sources: List[str] = field(default_factory=list)  # paper sections used


def keyword_intent(text: str) -> str:
    """Rule-based fallback used if the LLM router fails or returns an unknown label."""
    t = text.lower()
    if any(w in t for w in ("slide", "ppt", "presentation", "deck")):
        return "slides"
    if any(w in t for w in ("viva", "examiner", "mock question", "practice question", "interview")):
        return "viva"
    if any(w in t for w in ("summar", "overview", "tl;dr")):
        return "summary"
    if any(w in t for w in ("explain", "what is", "what are", "how does", "define", "meaning of", "why ")):
        return "concept"
    return "chat"


def _as_list(data: Any, key: str) -> list:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get(key), list):
            return data[key]
        for v in data.values():
            if isinstance(v, list):
                return v
    return []


def _clean_slides(raw: list) -> list:
    slides = []
    for i, s in enumerate(raw, 1):
        if not isinstance(s, dict):
            continue
        bullets = s.get("bullets", [])
        if isinstance(bullets, str):
            bullets = [bullets]
        slides.append({
            "title": str(s.get("title") or f"Slide {i}"),
            "bullets": [str(b) for b in bullets],
            "notes": str(s.get("notes") or ""),
        })
    return slides


def _clean_questions(raw: list) -> list:
    out = []
    for q in raw:
        if isinstance(q, dict) and q.get("question"):
            out.append({
                "difficulty": str(q.get("difficulty", "medium")).lower(),
                "question": str(q["question"]),
                "model_answer": str(q.get("model_answer") or q.get("answer") or ""),
            })
    return out


class PaperAgent:
    """Holds the chains for all agents and routes each user message to the right one."""

    def __init__(self, llm, use_reflection: bool = False):
        self.use_reflection = use_reflection
        text = StrOutputParser()
        retry = dict(
            retry_if_exception_type=(OutputParserException,),
            stop_after_attempt=2,
            wait_exponential_jitter=False,
        )
        # --- LCEL chains: prompt | model | parser ---
        self.router_chain = P.ROUTER | llm | text
        self.decide_chain = (P.DECIDE | llm | JsonOutputParser()).with_retry(**retry)
        self.summary_chain = P.SUMMARY | llm | text
        self.reflect_chain = P.REFLECT | llm | text
        self.extract_chain = (P.EXTRACT | llm | JsonOutputParser()).with_retry(**retry)
        self.explain_chain = P.EXPLAIN | llm | text
        self.slides_chain = (P.SLIDES | llm | JsonOutputParser()).with_retry(**retry)
        self.viva_chain = (P.VIVA | llm | JsonOutputParser()).with_retry(**retry)
        self.chat_chain = P.CHAT | llm | text

    # ------------------------------------------------------------------ intent detection
    def route(self, question: str, history: str, last_task: Optional[str]) -> str:
        try:
            raw = self.router_chain.invoke({
                "question": question,
                "history": history or "(none)",
                "last_task": last_task or "none",
            }).strip().lower()
            label = re.sub(r"[^a-z]", "", raw.split()[0]) if raw else ""
            if label in INTENTS:
                return label
        except Exception:
            pass
        return keyword_intent(question)

    # ------------------------------------------------------------------ agentic RAG decision
    def decide_retrieval(self, question: str, history: str):
        """The agent itself decides whether retrieval is needed and writes the search query."""
        try:
            out = self.decide_chain.invoke({"question": question, "history": history or "(none)"})
            if isinstance(out, dict):
                flag = out.get("retrieve", True)
                if isinstance(flag, str):
                    flag = flag.strip().lower() == "true"
                return bool(flag), (out.get("query") or question)
        except Exception:
            pass
        return True, question

    # ------------------------------------------------------------------ main entry point
    def run(self, question: str, index: PaperIndex, history: str = "", last_task: Optional[str] = None,
            n_slides: int = 8, n_questions: int = 8) -> Result:
        intent = self.route(question, history, last_task)
        hist = history or "(none)"

        if intent == "summary":
            docs = overview_docs(index)
            ctx = format_context(docs)
            summary = self.summary_chain.invoke({"title": index.title, "context": ctx})
            if self.use_reflection:
                summary = self.reflect_chain.invoke({"context": ctx, "draft": summary})
            return Result(intent, True, "whole paper overview", "text", summary,
                          "Gave a sectioned summary of the paper: " + summary[:300], sources_of(docs))

        if intent in ("slides", "viva"):
            docs = overview_docs(index)
            ctx = format_context(docs)
            points = self.extract_chain.invoke({"title": index.title, "context": ctx})
            key_points = json.dumps(points)[:6000]
            if intent == "slides":
                raw = self.slides_chain.invoke({
                    "title": index.title, "n_slides": n_slides, "key_points": key_points, "context": ctx})
                slides = _clean_slides(_as_list(raw, "slides"))
                if not slides:
                    raise ValueError("The model returned no slides. Please try again.")
                titles = ", ".join(s["title"] for s in slides)
                return Result(intent, True, "whole paper overview", "slides", slides,
                              f"Generated {len(slides)} slides: {titles}", sources_of(docs))
            raw = self.viva_chain.invoke({
                "title": index.title, "n_questions": n_questions, "key_points": key_points, "context": ctx})
            questions = _clean_questions(_as_list(raw, "questions"))
            if not questions:
                raise ValueError("The model returned no viva questions. Please try again.")
            return Result(intent, True, "whole paper overview", "viva", questions,
                          f"Generated {len(questions)} viva questions, e.g. " + questions[0]["question"],
                          sources_of(docs))

        # concept explanation and general chat: the agent decides whether to retrieve
        retrieve_flag, query = self.decide_retrieval(question, history)
        docs = retrieve(index, query, k=5) if retrieve_flag else []
        ctx = format_context(docs)
        chain = self.explain_chain if intent == "concept" else self.chat_chain
        answer = chain.invoke({"history": hist, "context": ctx, "question": question})
        return Result(intent, retrieve_flag, query if retrieve_flag else None, "text", answer,
                      answer[:300], sources_of(docs))
