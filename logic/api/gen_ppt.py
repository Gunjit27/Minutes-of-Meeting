# api/ppt_routes.py

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from logic.create_ppt.ppt_gen import run_ppt_generation

router = APIRouter()


@router.post("/generate-ppt")
async def generate_ppt():
    try:
        filename = run_ppt_generation()

        return FileResponse(
            path=filename,
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            filename="meeting_summary.pptx"
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))