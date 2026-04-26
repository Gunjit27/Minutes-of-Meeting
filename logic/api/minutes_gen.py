# api/mom_routes.py

from fastapi import APIRouter, HTTPException

from logic.mom.generate_mom import generate_minutes_of_meeting

router = APIRouter()


@router.post("/generate-mom")
async def generate_mom():
    try:
        mom = generate_minutes_of_meeting()

        return {
            "status": "success",
            "minutes_of_meeting": mom
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))