"""Retrieval over one meeting's chunks.

Three modes, selected with RETRIEVAL_MODE (default hybrid_rerank):
- vector:        the original setup, MMR over Chroma embeddings (k=3, fetch_k=10)
- hybrid:        BM25 + vector search, merged with reciprocal rank fusion
- hybrid_rerank: hybrid candidates re-scored by a cross-encoder, top k kept

Meeting transcripts are full of exact names, numbers and product terms
("Q3", "$4.2 billion", "EBITDA") that embeddings blur together, which is
what BM25 catches. The cross-encoder then reads query and chunk together,
which is more accurate than comparing two independently made vectors.
"""
import os
import uuid
from dataclasses import dataclass
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from logic.llm import get_embeddings

MODES = ("vector", "hybrid", "hybrid_rerank")
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid_rerank")
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

TOP_K = 3
CANDIDATES = 10  # per retriever, before fusion
RRF_K = 60  # standard reciprocal rank fusion constant


@dataclass
class MeetingIndex:
    vectorstore: Chroma
    bm25: BM25Retriever


def build_index(chunks: list[str], embeddings=None) -> MeetingIndex:
    docs = [Document(page_content=c, metadata={"chunk_id": i}) for i, c in enumerate(chunks)]

    # A fresh in-memory collection per meeting. The old code appended every
    # upload to one persisted collection, so answers mixed in earlier meetings.
    vectorstore = Chroma(
        collection_name=f"meeting_{uuid.uuid4().hex}",
        embedding_function=embeddings or get_embeddings(),
    )
    vectorstore.add_documents(docs)

    bm25 = BM25Retriever.from_documents(docs, k=CANDIDATES)
    return MeetingIndex(vectorstore=vectorstore, bm25=bm25)


def _vector_mmr(index: MeetingIndex, query: str, k: int) -> list[Document]:
    return index.vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=CANDIDATES, lambda_mult=0.5)


def _hybrid(index: MeetingIndex, query: str, k: int) -> list[Document]:
    vector_hits = index.vectorstore.similarity_search(query, k=CANDIDATES)
    bm25_hits = index.bm25.invoke(query)

    scores: dict[int, float] = {}
    docs: dict[int, Document] = {}
    for hits in (vector_hits, bm25_hits):
        for rank, doc in enumerate(hits):
            cid = doc.metadata["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            docs[cid] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[cid] for cid in ranked[:k]]


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


def _hybrid_rerank(index: MeetingIndex, query: str, k: int) -> list[Document]:
    candidates = _hybrid(index, query, k=2 * CANDIDATES)
    scores = _reranker().predict([(query, d.page_content) for d in candidates])
    ranked = sorted(zip(scores, candidates), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:k]]


_RETRIEVERS = {"vector": _vector_mmr, "hybrid": _hybrid, "hybrid_rerank": _hybrid_rerank}


def retrieve(index: MeetingIndex, query: str, mode: str = RETRIEVAL_MODE, k: int = TOP_K) -> list[str]:
    if mode not in _RETRIEVERS:
        raise ValueError(f"Unknown retrieval mode {mode!r}, expected one of {MODES}")
    return [doc.page_content for doc in _RETRIEVERS[mode](index, query, k)]
