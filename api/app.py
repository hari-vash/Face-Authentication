# # app.py
# import os
# from contextlib import asynccontextmanager
# from fastapi import FastAPI
# from logger.logger import logger
# from middleware.middleware import add_middleware, add_exception_handlers

# # Import routers
# from routes.home import router as home_router
# from routes.health import router as health_router
# from api.routes.verification_fixed import router as verification_router
# from routes.embedding import router as embedding_router

# # Lifespan manager
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Manage application lifecycle events."""
#     logger.info("Starting Face Recognition API application")
#     try:
#         # Initialize any required resources here
#         logger.info("Initializing DeepFace models...")
        
#         # You can pre-load DeepFace models here for better performance
#         # This will download models on first run
#         from deepface import DeepFace
        
#         # Pre-load the model to avoid first-time delays
#         try:
#             import numpy as np
#             dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
#             DeepFace.represent(img_path=dummy_image, model_name='Facenet', enforce_detection=False)
#             logger.info("DeepFace models loaded successfully")
#         except Exception as e:
#             logger.warning(f"Could not pre-load DeepFace models: {str(e)}")
        
#         logger.info("Application startup completed successfully")
#     except Exception as e:
#         logger.error(f"Failed to start application: {str(e)}")
#         raise RuntimeError(f"Application startup failed: {str(e)}")
    
#     yield
    
#     logger.info("Shutting down Face Recognition API application")

# # Initialize FastAPI app
# app = FastAPI(
#     title="Face Recognition API",
#     description="Industrial-level API for face verification and embedding generation using DeepFace",
#     version="1.0.0",
#     docs_url="/docs",
#     redoc_url="/redoc",
#     lifespan=lifespan
# )

# # Add middleware and exception handlers
# add_middleware(app)
# add_exception_handlers(app)

# # Include routes
# app.include_router(home_router)
# app.include_router(health_router)
# app.include_router(verification_router)
# app.include_router(embedding_router)
##################################################################################################################################################


# app.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from logger.logger import logger
from middleware.middleware import add_middleware, add_exception_handlers

# Import routers
from routes.home import router as home_router
from routes.health import router as health_router
from routes.verification_fixed import router as verification_router
from routes.embedding import router as embedding_router

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle events."""
    logger.info("Starting Face Recognition API application")
    try:
        # Initialize any required resources here
        logger.info("Initializing DeepFace models...")
        
        # You can pre-load DeepFace models here for better performance
        # This will download models on first run
        from deepface import DeepFace
        
        # Pre-load the model to avoid first-time delays
        try:
            import numpy as np
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            DeepFace.represent(img_path=dummy_image, model_name='Facenet', enforce_detection=False)
            logger.info("DeepFace models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not pre-load DeepFace models: {str(e)}")
        
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise RuntimeError(f"Application startup failed: {str(e)}")
    
    yield
    
    logger.info("Shutting down Face Recognition API application")

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Industrial-level API for face verification and embedding generation using DeepFace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware and exception handlers
add_middleware(app)
add_exception_handlers(app)

# Include routes
app.include_router(home_router)
app.include_router(health_router)
app.include_router(verification_router)
app.include_router(embedding_router)