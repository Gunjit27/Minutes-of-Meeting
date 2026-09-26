import state
from logic.rag.retrieval import retrieve
from logic.llm import get_llm

llm = get_llm(temperature=0.2)

prompt = """
You are an assistant for answering questions based on the context provided.
Use only the following context to answer the question.
If you don't know the answer, say you don't know.

Context:
{context}

Question:
{query}
"""


def load_vectorstore():
    vectorstore = state.get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vectorstore is not initialized.")
    return vectorstore


def query_vectorstore(query: str):
    results = retrieve(load_vectorstore(), query)

    context = "\n\n".join(results)
    return context



def generate_answer(query: str):
    context = query_vectorstore(query)

    formatted_prompt = prompt.format(
        context=context,
        query=query
    )

    response = llm.invoke(formatted_prompt)

    return response.content
