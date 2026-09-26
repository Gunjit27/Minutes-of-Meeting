import state
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logic.rag.retrieval import build_index

# Shared with evals/run.py, so the eval chunks transcripts exactly like the app.
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    add_start_index=True,
)


def chunk_transcript(transcript: str):
    texts = SPLITTER.split_text(transcript)
    return texts


def embed_chunks():
    chunks = chunk_transcript(state.get_transcript())
    index = build_index(chunks)
    state.set_vectorstore(index)
    return index
