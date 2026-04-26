# import asyncio
# import os
# import state

# from langchain_ollama import ChatOllama
# from langchain_community.vectorstores import Chroma
# from langchain_ollama import OllamaEmbeddings
# import edge_tts


# # ---------------- LLM ---------------- #

# llm = ChatOllama(
#     model='llama3.1:8b',
#     temperature=0.2
# )

# prompt = """
# You are an assistant for answering questions based on the context provided.
# Use only the following context to answer the question.
# If you don't know the answer, say you don't know.

# Context:
# {context}

# Question:
# {query}
# """


# # ---------------- VECTORSTORE ---------------- #

# def load_vectorstore():
#     vectorstore = state.get_vectorstore()
#     if vectorstore is None:
#         raise ValueError("Vectorstore is not initialized.")
#     return vectorstore


# def query_vectorstore(query: str):
#     vectorstore = load_vectorstore()

#     retriever = vectorstore.as_retriever(
#         search_type="mmr",
#         search_kwargs={
#             "k": 3,
#             "fetch_k": 10,
#             "lambda_mult": 0.5
#         }
#     )

#     results = retriever.invoke(query)

#     context = "\n\n".join([doc.page_content for doc in results])
#     return context


# # ---------------- EDGE TTS ---------------- #

# def generate_tts(text: str, filename="output.mp3"):
#     communicate = edge_tts.Communicate(
#         text,
#         voice="en-IN-NeerjaNeural",
#         rate="+30%"   # 🔥 faster speech
#     )
#     communicate.save(filename)


# def speak_text(text: str):
#     filename = "output.mp3"

#     # Clean text
#     text = text.replace("\n", " ").strip()

#     # Limit length (important)
#     text = text[:600]

#     try:
#         asyncio.run(generate_tts(text, filename))
#     except RuntimeError:
#         loop = asyncio.get_event_loop()
#         loop.run_until_complete(generate_tts(text, filename))

#     # Play audio (Windows)
#     # os.system(f"start {filename}")


# # ---------------- MAIN FUNCTION ---------------- #

# def generate_answer_tts(query: str):
#     context = query_vectorstore(query)

#     formatted_prompt = prompt.format(
#         context=context,
#         query=query
#     )

#     response = llm.invoke(formatted_prompt)

#     answer = response.content

#     # 🔊 Speak answer
#     speak_text(answer)

#     return answer

# services/qa_tts.py

import uuid
import state
import edge_tts

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
    return "\n\n".join([doc.page_content for doc in results])


# ---------------- EDGE TTS (ASYNC) ---------------- #

async def generate_tts(text: str, filename: str):
    communicate = edge_tts.Communicate(
        text,
        voice="en-IN-NeerjaNeural",
        rate="+30%"
    )
    await communicate.save(filename)


# ---------------- MAIN FUNCTION ---------------- #

async def generate_answer_tts(query: str):
    # 1️⃣ Retrieve context (sync, fast enough)
    context = query_vectorstore(query)

    # 2️⃣ Generate answer (sync LLM call)
    formatted_prompt = prompt.format(
        context=context,
        query=query
    )

    response = llm.invoke(formatted_prompt)
    answer = response.content

    # 3️⃣ Clean + limit text
    clean_answer = answer.replace("\n", " ").strip()[:600]

    # 4️⃣ Unique filename (IMPORTANT)
    filename = f"output_{uuid.uuid4()}.mp3"

    # 5️⃣ Async TTS (correct way)
    await generate_tts(clean_answer, filename)

    return {
        "answer": answer,
        "audio_file": filename
    }