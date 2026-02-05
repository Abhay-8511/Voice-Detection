import os
from fastapi import HTTPException

API_KEY = os.getenv("API_KEY")

def validate_api_key(x_api_key: str):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
