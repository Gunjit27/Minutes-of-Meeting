import uuid
import state
import edge_tts

from langchain_ollama import ChatOllama


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
    return "\n\n".join([doc.page_content for doc in results])



async def generate_tts(text: str, filename: str):
    communicate = edge_tts.Communicate(
        text,
        voice="en-IN-NeerjaNeural",
        rate="+30%"
    )
    await communicate.save(filename)



async def generate_answer_tts(query: str):
    context = query_vectorstore(query)

    formatted_prompt = prompt.format(
        context=context,
        query=query
    )

    response = llm.invoke(formatted_prompt)
    answer = response.content

    clean_answer = answer.replace("\n", " ").strip()[:600]

    filename = f"output_{uuid.uuid4()}.mp3"

    await generate_tts(clean_answer, filename)

    return {
        "answer": answer,
        "audio_file": filename
    }
