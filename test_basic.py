"""Offline tests with a fake LLM and fake vector store (no API key or internet needed).
Run:  python test_basic.py   (or: pytest)"""
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from agents import PaperAgent, keyword_intent
from pptx_builder import build_pptx
from rag import PaperIndex, split_into_sections

PAPER = """Some Title
Abstract
We study a thing.
1. Introduction
Things are hard.
2. Methodology
We do a method.
3 Results
It works well.
References
[1] Someone."""


class FakeStore:
    def similarity_search(self, query, k=4, filter=None):
        sec = (filter or {}).get("section", "methodology")
        return [Document(page_content=f"Text about {sec} #{i}", metadata={"section": sec, "chunk": i}) for i in range(k)]


def fake_llm(prompt_value):
    t = prompt_value.to_string()
    if "Intent Router" in t:
        return AIMessage(content=keyword_intent(t.split("Latest user message:")[-1]))
    if "Retrieval Planner" in t:
        return AIMessage(content='```json\n{"retrieve": true, "query": "attention"}\n```')
    if "Summarizer Agent" in t:
        return AIMessage(content="## Overview\nA fake summary.")
    if "extraction mode" in t:
        return AIMessage(content='{"problem": "p", "contributions": ["c"], "methods": ["m"], "results": ["r"], "limitations": [], "keywords": []}')
    if "explanation mode" in t:
        return AIMessage(content="Attention is a fake explanation [methodology].")
    if "Slide-Generator" in t:
        return AIMessage(content='{"slides": [{"title": "Intro", "bullets": ["a", "b"], "notes": "n"}, {"title": "Method", "bullets": ["x"], "notes": ""}]}')
    if "Viva Agent" in t:
        return AIMessage(content='{"questions": [{"difficulty": "easy", "question": "Why?", "model_answer": "Because."}]}')
    return AIMessage(content="chat reply")


def make_agent():
    return PaperAgent(RunnableLambda(fake_llm)), PaperIndex(FakeStore(), "Fake Paper", ["abstract", "methodology"], 4)


def test_sections():
    names = [n for n, _ in split_into_sections(PAPER)]
    assert names == ["front matter", "abstract", "introduction", "methodology", "results"], names


def test_keyword_intent():
    assert keyword_intent("make a ppt") == "slides"
    assert keyword_intent("viva questions please") == "viva"
    assert keyword_intent("summarize it") == "summary"
    assert keyword_intent("explain attention") == "concept"
    assert keyword_intent("thanks") == "chat"


def test_routing_and_outputs():
    agent, index = make_agent()
    r = agent.run("Summarize the paper", index)
    assert (r.intent, r.kind, r.retrieved) == ("summary", "text", True)
    r = agent.run("Make slides", index, n_slides=2)
    assert r.intent == "slides" and len(r.data) == 2
    r = agent.run("Give me viva questions", index)
    assert r.intent == "viva" and r.data[0]["question"] == "Why?"
    r = agent.run("Explain attention", index)
    assert r.intent == "concept" and r.retrieved and r.query == "attention"


def test_reflection_toggle():
    agent, index = make_agent()
    agent.use_reflection = True
    assert agent.run("Summarize the paper", index).kind == "text"


def test_pptx():
    data = build_pptx("Deck", [{"title": "A", "bullets": ["1"], "notes": "n"}, {"title": "B", "bullets": ["2", "3"], "notes": ""}])
    assert data[:2] == b"PK" and len(data) > 5000


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
