# api/qa_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import FileResponse

from logic.rag.query import generate_answer
from logic.rag.tts_query import generate_answer_tts

router = APIRouter()


class QueryRequest(BaseModel):
    query: str


# ---------------- TEXT ONLY ---------------- #

@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        answer = generate_answer(request.query)

        return {
            "status": "success",
            "answer": answer
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- TTS (ANSWER + AUDIO FILE) ---------------- #

@router.post("/ask-tts")
async def ask_question_tts(request: QueryRequest):
    try:
        result = await generate_answer_tts(request.query)

        return {
            "status": "success",
            "answer": result["answer"],
            "audio_file": result["audio_file"]
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------- OPTIONAL: STREAM AUDIO ---------------- #

@router.get("/audio/{filename}")
async def get_audio(filename: str):
    return FileResponse(
        path=filename,
        media_type="audio/mpeg",
        filename="response.mp3"
    )