import state
from langchain_text_splitters import RecursiveCharacterTextSplitter

from logic.rag.retrieval import build_index


def chunk_transcript(transcript: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200, 
    )
    texts = text_splitter.split_text(transcript)
    return texts


def embed_chunks():
    chunks = chunk_transcript(state.get_transcript())
    index = build_index(chunks)
    state.set_vectorstore(index)
    return index
