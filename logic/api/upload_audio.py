from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
import uuid

from logic.transcription.transcribe_audio import transcribe_audio
from logic.transcription.clean_transcript import clean_transcript
from logic.rag.chunk_embed import embed_chunks

router = APIRouter()

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload_audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith((".mp3", ".wav", ".m4a")):
            raise HTTPException(status_code=400, detail="Unsupported file format")

        file_id = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, file_id)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        raw_transcript = transcribe_audio(file_path)

        cleaned_transcript = clean_transcript(raw_transcript)

        os.remove(file_path)
        embed_chunks()
        return {
            "status": "success",
            "transcript": cleaned_transcript
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
