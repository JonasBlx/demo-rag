"""RAG chain, broken into explicit steps so the UI can show everything
(retrieved chunks, scores, the context sent to the model, timings, tokens)."""

from collections import Counter

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

from src.config import PERSIST_DIR, RETRIEVAL_K, get_embeddings

PROMPT = ChatPromptTemplate.from_template(
    """You are an expert assistant for LangChain. Answer the question using ONLY
the context below. If the information is not in the context, say so explicitly —
never make it up. Cite the sources (URLs) you rely on.

Context:
{context}

Question: {question}

Answer:"""
)


def load_vectorstore() -> Chroma:
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=get_embeddings())


def retrieve(
    vectorstore: Chroma,
    question: str,
    k: int = RETRIEVAL_K,
    fetch_k: int = 60,
    per_source: int = 2,
):
    """Return up to `k` (Document, relevance score 0–1, higher = closer), diversified.

    Pure top-k similarity lets one dominant page (e.g. the RAG tutorial) fill every
    slot and crowd out the concept pages. So we fetch a larger pool by similarity,
    then keep at most `per_source` chunks per source URL — surfacing other pages
    while preserving the scores (unlike MMR). Ranking order is preserved.
    """
    pool = vectorstore.similarity_search_with_relevance_scores(question, k=fetch_k)

    keep, seen = [], Counter()
    for i, (doc, _score) in enumerate(pool):  # pass 1: cap per source
        src = doc.metadata.get("source")
        if seen[src] < per_source:
            keep.append(i)
            seen[src] += 1
            if len(keep) >= k:
                break
    for i in range(len(pool)):  # pass 2: top up by rank if still short
        if len(keep) >= k:
            break
        if i not in keep:
            keep.append(i)

    return [pool[i] for i in sorted(keep)]


def build_context(scored_docs) -> str:
    """Assemble the context sent to the model (shown in the UI)."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc, _score in scored_docs
    )


def generate(llm, question: str, context: str):
    """Call Claude and return the AIMessage (which also carries usage_metadata)."""
    return llm.invoke(PROMPT.format_messages(context=context, question=question))
