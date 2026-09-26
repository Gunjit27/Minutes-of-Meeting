"""Model factory, so the same code runs on local Ollama or on hosted Groq.

LLM_PROVIDER=ollama (default) uses llama3.1:8b through a local Ollama server.
LLM_PROVIDER=groq uses Groq's hosted openai/gpt-oss-20b (needs GROQ_API_KEY).

EMBED_PROVIDER=ollama (default) uses nomic-embed-text through Ollama.
EMBED_PROVIDER=hf runs a sentence-transformers model in-process, for hosts without Ollama.
"""
import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")


def get_llm(temperature: float = 0.2):
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"), temperature=temperature)

    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"), temperature=temperature)


def get_embeddings():
    if EMBED_PROVIDER == "hf":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
