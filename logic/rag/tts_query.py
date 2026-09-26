import uuid
import state
import edge_tts

from logic.rag.query import llm, prompt, query_vectorstore


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
