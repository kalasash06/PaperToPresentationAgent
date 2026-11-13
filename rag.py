"""PDF loading, section-aware chunking and the ChromaDB vector store (the RAG layer)."""
import io
import math
import re
import uuid
from dataclasses import dataclass
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
OVERVIEW_QUERY = "main points, contributions, method and findings"

SECTION_NAMES = [
    "abstract", "introduction", "related work", "background", "literature review",
    "methodology", "methods", "method", "proposed method", "approach",
    "experiments", "experimental setup", "results", "results and discussion",
    "discussion", "conclusion", "conclusions", "future work", "references",
]
_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+|[IVX]+\.\s+)?("
    + "|".join(sorted(map(re.escape, SECTION_NAMES), key=len, reverse=True))
    + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


class ChromaDefaultEmbeddings(Embeddings):
    """Wraps Chroma's built-in MiniLM (ONNX) embedding model. Free, local, no API key."""

    def __init__(self):
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [[float(x) for x in vec] for vec in self._fn(texts)]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    return ChromaDefaultEmbeddings()


@dataclass
class PaperIndex:
    store: object          # Chroma vector store (anything with .similarity_search works)
    title: str
    sections: List[str]    # section names in reading order
    n_chunks: int


def split_into_sections(text: str):
    """Split paper text on common headings (Abstract, Introduction, Method...). References are dropped."""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("full text", text)]
    sections = []
    front = text[: matches[0].start()].strip()
    if front:
        sections.append(("front matter", front))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        name = m.group(1).strip().lower()
        body = text[m.end():end].strip()
        if name == "references" or not body:
            continue
        sections.append((name, body))
    return sections or [("full text", text)]


def _guess_title(reader: PdfReader, text: str, filename: str) -> str:
    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        meta_title = ""
    if len(meta_title) > 5:
        return meta_title[:150]
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 8:
            return line[:150]
    return filename


def index_pdf(data: bytes, filename: str = "paper.pdf") -> PaperIndex:
    """Read the PDF, split into section-tagged chunks, embed and store them in ChromaDB."""
    reader = PdfReader(io.BytesIO(data))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    text = re.sub(r"-\n(\w)", r"\1", text)  # fix words hyphenated across lines
    if len(text.strip()) < 200:
        raise ValueError("No readable text found. The PDF may be a scanned image (needs OCR).")

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs, order = [], []
    for name, body in split_into_sections(text):
        if name not in order:
            order.append(name)
        for j, chunk in enumerate(splitter.split_text(body)):
            docs.append(Document(page_content=chunk, metadata={"section": name, "chunk": j}))

    store = Chroma.from_documents(
        docs,
        embedding=get_embeddings(),
        collection_name="paper_" + uuid.uuid4().hex[:8],
    )
    return PaperIndex(store=store, title=_guess_title(reader, text, filename), sections=order, n_chunks=len(docs))


def retrieve(index: PaperIndex, query: str, k: int = 5) -> List[Document]:
    """Similarity search over the paper chunks."""
    return index.store.similarity_search(query, k=k)


def overview_docs(index: PaperIndex, target: int = 14) -> List[Document]:
    """Cover the whole paper: pull the most relevant chunks from every section."""
    names = [s for s in index.sections if s != "front matter"] or index.sections
    per = max(2, math.ceil(target / len(names)))
    seen, out = set(), []
    for name in names:
        for d in index.store.similarity_search(OVERVIEW_QUERY, k=per, filter={"section": name}):
            if d.page_content not in seen:
                seen.add(d.page_content)
                out.append(d)
    return out


def format_context(docs: List[Document]) -> str:
    if not docs:
        return "(no excerpts retrieved)"
    return "\n\n---\n\n".join(f"[{d.metadata.get('section', '?')}]\n{d.page_content}" for d in docs)


def sources_of(docs: List[Document]) -> List[str]:
    out = []
    for d in docs:
        s = d.metadata.get("section", "?")
        if s not in out:
            out.append(s)
    return out
