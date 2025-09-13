# routes/health.py
from fastapi import APIRouter
from fastapi.responses import JSONResponse
import time
import psutil
import os

router = APIRouter(prefix="/health", tags=["Health Check"])

@router.get("/")
async def health_check():
    """
    Health check endpoint to verify API status and system resources.
    """
    try:
        # Get system information
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uptime": time.time(),
            "system_info": {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "disk_usage_percent": disk.percent,
                "disk_free_gb": round(disk.free / (1024**3), 2)
            },
            "deepface_status": "available",
            "api_version": "1.0.0"
        })
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

@router.get("/simple")
async def simple_health_check():
    """
    Simple health check endpoint.
    """
    return JSONResponse(content={
        "status": "ok",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    })