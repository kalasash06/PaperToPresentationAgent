"""Streamlit GUI for the Paper-to-Presentation & Viva Agent.  Run with:  streamlit run app.py"""
import hashlib
from html import escape

import streamlit as st

import prompts as P
from agents import PaperAgent
from config import get_llm, llm_status
from pptx_builder import build_pptx
from rag import index_pdf

st.set_page_config(page_title="Paper-to-Presentation & Viva Agent", page_icon="🎓", layout="wide")

CSS = """
<style>
.block-container { padding-top: 2.2rem; max-width: 1100px; }
#MainMenu, footer { visibility: hidden; }

/* hero */
.hero { padding: 1.6rem 1.8rem; border-radius: 20px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, rgba(139,92,246,.22), rgba(56,189,248,.14));
        border: 1px solid rgba(139,92,246,.35); }
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800; letter-spacing: -.5px;
           background: linear-gradient(90deg, #C4B5FD, #67E8F9);
           -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.hero p { margin: .35rem 0 0; color: #AEB4C6; font-size: 1.02rem; }

/* feature + stat cards */
.card { background: #161923; border: 1px solid #262B3A; border-radius: 16px;
        padding: 1.1rem 1.2rem; height: 100%; }
.card .ico { font-size: 1.7rem; }
.card h4 { margin: .4rem 0 .25rem; font-size: 1.05rem; }
.card p { margin: 0; color: #9AA1B5; font-size: .92rem; line-height: 1.45; }
.stat { background: #161923; border: 1px solid #262B3A; border-radius: 14px; padding: .8rem 1rem; }
.stat .v { font-size: 1.5rem; font-weight: 700; color: #C4B5FD; }
.stat .l { font-size: .78rem; color: #8A91A6; text-transform: uppercase; letter-spacing: .06em; }
.paper { background: #161923; border: 1px solid #262B3A; border-left: 4px solid #8B5CF6;
         border-radius: 12px; padding: .8rem 1.1rem; margin-bottom: .8rem; }
.paper b { color: #E6E8EF; }
.paper span { color: #8A91A6; font-size: .85rem; }

/* badges under answers */
.badges { display: flex; flex-wrap: wrap; gap: .4rem; margin-bottom: .6rem; }
.badge { font-size: .74rem; padding: .18rem .6rem; border-radius: 999px;
         border: 1px solid #343A4D; background: #1C2030; color: #B7BDD0; }
.badge.intent { background: rgba(139,92,246,.18); border-color: rgba(139,92,246,.5); color: #DDD6FE; font-weight: 600; }
.badge.ok { background: rgba(52,211,153,.12); border-color: rgba(52,211,153,.4); color: #A7F3D0; }
.badge.skip { background: rgba(250,204,21,.10); border-color: rgba(250,204,21,.35); color: #FDE68A; }

/* buttons */
.stButton > button, .stDownloadButton > button { border-radius: 12px; font-weight: 600; padding: .6rem 1rem;
        border: 1px solid #343A4D; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #8B5CF6; transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(139,92,246,.25); }
[data-testid="stSidebar"] { border-right: 1px solid #1F2433; }
[data-testid="stExpander"] { border-radius: 12px; border-color: #262B3A; }
[data-testid="stChatMessage"] { background: transparent; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

for key, default in {"messages": [], "index": None, "file_hash": None, "last_task": None, "agent": None}.items():
    st.session_state.setdefault(key, default)


def history_text(messages, n=6):
    """Conversation memory: the last few turns as plain text for the prompts."""
    lines = []
    for m in messages[-n:]:
        who = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{who}: {m.get('memo') or m['content']}")
    return "\n".join(lines)


def friendly_error(e: Exception) -> str:
    msg = str(e)
    if "NOT_FOUND" in msg or "404" in msg:
        return ("The model name was rejected. Set `GEMINI_MODEL` in your `.env` to a current model "
                "(see https://aistudio.google.com) and restart the app.\n\n" + msg[:300])
    if "API key" in msg or "API_KEY" in msg or "PERMISSION_DENIED" in msg or "401" in msg:
        return "The API key looks wrong or missing. Check `GOOGLE_API_KEY` in your `.env` and restart.\n\n" + msg[:300]
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Rate limit reached on the free tier. Wait a minute and try again."
    return msg


# ------------------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎓 Paper Agent")
    st.caption("Summary · Slides · Concepts · Viva")
    st.divider()
    uploaded = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

    st.markdown("#### Settings")
    n_slides = st.slider("Number of slides", 5, 15, 8)
    n_questions = st.slider("Number of viva questions", 5, 15, 8)
    use_reflection = st.checkbox("Self-check summary (reflection)", value=False,
                                 help="Adds a fact-checking pass over the summary. Slower, fewer unsupported claims.")

    provider, model, key_ok = llm_status()
    st.caption(f"🤖 LLM: `{provider}` / `{model}`")
    if not key_ok:
        st.warning("API key not found. Copy .env.example to .env and add your key.")

    with st.expander("Prompts used (system + user)"):
        for name, (system_prompt, user_prompt) in P.REGISTRY.items():
            st.markdown(f"**{name}**")
            st.caption("System prompt")
            st.code(system_prompt, language="text")
            st.caption("User prompt")
            st.code(user_prompt, language="text")

# ------------------------------------------------------------------------------ index the paper
if uploaded is not None:
    data = uploaded.getvalue()
    digest = hashlib.md5(data).hexdigest()
    if digest != st.session_state.file_hash:
        with st.spinner("Reading the paper and building the vector index (first run downloads a small embedding model)..."):
            try:
                st.session_state.index = index_pdf(data, uploaded.name)
                st.session_state.file_hash = digest
                st.session_state.messages = []
                st.session_state.last_task = None
            except Exception as e:
                st.session_state.index = None
                st.error(f"Could not process this PDF: {e}")

index = st.session_state.index

# ------------------------------------------------------------------------------ header
st.markdown(
    """<div class="hero"><h1>Paper-to-Presentation &amp; Viva Agent</h1>
    <p>Turn any research paper into a summary, a slide deck and viva practice, grounded in the paper itself.</p></div>""",
    unsafe_allow_html=True,
)

if index is None:
    cols = st.columns(3)
    cards = [
        ("📝", "Summary", "A sectioned overview: problem, method, results and conclusion."),
        ("🎞️", "Slides", "A presentation-ready deck with speaker notes, downloadable as .pptx."),
        ("🎤", "Viva practice", "Likely examiner questions of mixed difficulty with model answers."),
    ]
    for col, (ico, title, text) in zip(cols, cards):
        col.markdown(f'<div class="card"><div class="ico">{ico}</div><h4>{title}</h4><p>{text}</p></div>',
                     unsafe_allow_html=True)
    st.markdown("")
    st.info("👈 Upload a research paper PDF in the sidebar to begin.")
else:
    st.markdown(f'<div class="paper"><span>Current paper</span><br><b>{escape(index.title)}</b></div>',
                unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    s1.markdown(f'<div class="stat"><div class="v">{index.n_chunks}</div><div class="l">Chunks in ChromaDB</div></div>',
                unsafe_allow_html=True)
    s2.markdown(f'<div class="stat"><div class="v">{len(index.sections)}</div><div class="l">Sections detected</div></div>',
                unsafe_allow_html=True)
    s3.markdown(f'<div class="stat"><div class="v">{len(st.session_state.messages) // 2}</div><div class="l">Tasks done</div></div>',
                unsafe_allow_html=True)
    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("📝 Summarize", use_container_width=True):
        st.session_state.pending = "Give me a sectioned summary of the paper."
    if c2.button("🎞️ Make slides", use_container_width=True):
        st.session_state.pending = f"Create a {n_slides} slide presentation for this paper."
    if c3.button("🎤 Viva questions", use_container_width=True):
        st.session_state.pending = f"Give me {n_questions} likely viva questions with model answers."
    if c4.button("🧹 Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_task = None
        st.rerun()
    st.markdown("")

pending = st.session_state.pop("pending", None)
prompt = st.chat_input("Ask for a summary, slides, a concept explained, or viva practice...", disabled=index is None)
prompt = prompt or pending

# ------------------------------------------------------------------------------ run the agents
if prompt and index is not None:
    past = list(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        if st.session_state.agent is None:
            st.session_state.agent = PaperAgent(get_llm())
        agent = st.session_state.agent
        agent.use_reflection = use_reflection
        with st.spinner("Agents are working..."):
            result = agent.run(
                prompt, index, history=history_text(past), last_task=st.session_state.last_task,
                n_slides=n_slides, n_questions=n_questions,
            )
        if result.intent != "chat":
            st.session_state.last_task = result.intent
        st.session_state.messages.append({
            "role": "assistant", "content": result.memo, "memo": result.memo,
            "kind": result.kind, "data": result.data,
            "meta": {"intent": result.intent, "retrieved": result.retrieved,
                     "query": result.query, "sources": result.sources},
        })
    except Exception as e:
        st.session_state.messages.pop()  # remove the user turn that failed
        st.error(f"Something went wrong: {friendly_error(e)}")

# ------------------------------------------------------------------------------ render the chat
DIFF_ICON = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
INTENT_ICON = {"summary": "📝", "slides": "🎞️", "viva": "🎤", "concept": "💡", "chat": "💬"}


def badges(meta):
    intent = str(meta.get("intent", ""))
    parts = [f'<span class="badge intent">{INTENT_ICON.get(intent, "")} {escape(intent)}</span>']
    if meta.get("retrieved"):
        q = escape(str(meta.get("query") or ""))
        parts.append(f'<span class="badge ok">🔎 retrieved: {q}</span>')
    else:
        parts.append('<span class="badge skip">⚡ answered from memory (no retrieval)</span>')
    for s in meta.get("sources") or []:
        parts.append(f'<span class="badge">§ {escape(str(s))}</span>')
    return '<div class="badges">' + "".join(parts) + "</div>"


def render(i, m):
    with st.chat_message(m["role"], avatar="🧑‍🎓" if m["role"] == "user" else "🎓"):
        if m["role"] == "user":
            st.markdown(m["content"])
            return
        st.markdown(badges(m.get("meta", {})), unsafe_allow_html=True)
        kind = m.get("kind", "text")
        if kind == "text":
            st.markdown(m["data"])
        elif kind == "slides":
            for n, s in enumerate(m["data"], 1):
                with st.expander(f"Slide {n}: {s['title']}", expanded=(n == 1)):
                    for b in s["bullets"]:
                        st.markdown(f"- {b}")
                    if s["notes"]:
                        st.caption(f"🗒️ Speaker notes: {s['notes']}")
            st.download_button(
                "⬇️ Download .pptx", data=build_pptx(index.title if index else "Presentation", m["data"]),
                file_name="presentation.pptx",
                mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                key=f"pptx_{i}",
            )
        elif kind == "viva":
            for n, q in enumerate(m["data"], 1):
                icon = DIFF_ICON.get(q["difficulty"], "⚪")
                with st.expander(f"{icon} Q{n} · {q['difficulty']} · {q['question']}"):
                    st.markdown(q["model_answer"])


for i, m in enumerate(st.session_state.messages):
    render(i, m)