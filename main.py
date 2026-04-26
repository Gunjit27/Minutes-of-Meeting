from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from logic.api.upload_audio import router as upload_router        # /upload_audio
from logic.api.ask import router as qa_router         # /ask, /ask-tts, /audio
from logic.api.minutes_gen import router as mom_router       # /generate-mom
from logic.api.gen_ppt import router as ppt_router       # /generate-ppt


app = FastAPI(
    title="Meeting Intelligence API",
    description="Audio → Transcript → Q&A → MoM → PPT → TTS",
    version="1.0.0"
)


# ---------------- CORS ---------------- #

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ok for personal project
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- ROUTERS ---------------- #

app.include_router(upload_router, tags=["Upload"])
app.include_router(qa_router, tags=["Q&A"])
app.include_router(mom_router, tags=["MoM"])
app.include_router(ppt_router, tags=["PPT"])


# ---------------- ROOT ---------------- #

@app.get("/")
async def root():
    return {
        "message": "Meeting Intelligence API is running 🚀",
        "endpoints": {
            "upload_audio": "/upload_audio",
            "ask": "/ask",
            "ask_tts": "/ask-tts",
            "generate_mom": "/generate-mom",
            "generate_ppt": "/generate-ppt"
        }
    }