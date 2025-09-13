# routes/home.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import time

router = APIRouter(tags=["Home"])

@router.get("/")
async def root():
    """
    Root endpoint - Welcome message and API information.
    """
    return JSONResponse(content={
        "message": "Welcome to Face Recognition API",
        "description": "API for face verification and embedding generation using DeepFace",
        "version": "1.0.0",
        "endpoints": {
            "verification": "/verification/verify - POST",
            "embedding_generation": "/embedding/generate - POST",
            "health_check": "/health - GET",
            "api_docs": "/docs",
            "redoc_docs": "/redoc"
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "operational"
    })