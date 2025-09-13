# server.py
import os
import uvicorn
from logger.logger import logger

if __name__ == "__main__":
    try:
        logger.info("Starting Face Recognition API server...")
        
        # Import the app
        from app import app
        
        # Server configuration
        host = os.environ.get("HOST", "0.0.0.0")
        port = int(os.environ.get("PORT", 8000))
        
        logger.info(f"Server will start on {host}:{port}")
        
        # Run the FastAPI server
        uvicorn.run(
            app,  # Pass the app object directly
            host=host,
            port=port,
            log_level="info",
            access_log=True,
            reload=False  # Set to True for development
        )
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        exit(1)