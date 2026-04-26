from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import state
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_transcript(transcript: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200, 
    )
    texts = text_splitter.split_text(transcript)
    return texts


def embed_chunks():
    chunks = chunk_transcript(state.get_transcript())
    embeddings = OllamaEmbeddings(model='nomic-embed-text')

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    vectorstore.add_texts(chunks)
    # vectorstore.persist()

    state.set_vectorstore(vectorstore)

    return vectorstore
