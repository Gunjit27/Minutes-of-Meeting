# services/qa.py

import state
from langchain_ollama import ChatOllama


# ---------------- LLM ---------------- #

llm = ChatOllama(
    model='llama3.1:8b',
    temperature=0.2
)

prompt = """
You are an assistant for answering questions based on the context provided.
Use only the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{query}
"""


# ---------------- VECTORSTORE ---------------- #

def load_vectorstore():
    vectorstore = state.get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized.")
    return vectorstore


def query_vectorstore(query: str):
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
            "lambda_mult": 0.5
        }
    )

    results = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in results])
    return context


# ---------------- MAIN FUNCTION ---------------- #

def generate_answer(query: str):
    context = query_vectorstore(query)

    formatted_prompt = prompt.format(
        context=context,
        query=query
    )

    response = llm.invoke(formatted_prompt)

    return response.content