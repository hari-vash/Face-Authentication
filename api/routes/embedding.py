# routes/embedding.py
import numpy as np
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from deepface import DeepFace

from schema.output_schema import EmbeddingResponse, ErrorResponse
from logger.logger import logger

router = APIRouter(prefix="/embedding", tags=["Face Embedding Generation"])

def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess uploaded image for DeepFace processing.
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embedding(
    sid: str = Form(..., description="Student ID or User ID"),
    photo: UploadFile = File(..., description="Photo for embedding generation")
):
    """
    Generate face embedding from uploaded photo.
    
    - **sid**: Student ID or User ID
    - **photo**: Photo file for embedding generation
    """
    try:
        logger.info(f"Starting embedding generation for SID: {sid}")
        
        # Validate image file
        if not photo.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and preprocess image
        image_data = await photo.read()
        img_array = preprocess_image(image_data)
        
        # Generate embedding using DeepFace
        try:
            # Generate embedding using DeepFace
            result = DeepFace.represent(
                img_path=img_array,
                model_name='VGG-Face',  # You can change to 'VGG-Face', 'ArcFace', etc.
                enforce_detection=True
            )
            
            # Extract embedding from result
            embedding = result[0]["embedding"]
            
            logger.info(f"Successfully generated embedding for SID: {sid}")
            
            return EmbeddingResponse(
                sid=sid,
                embedding=embedding,
                message="Embedding generated successfully"
            )
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for SID {sid}: {str(e)}")
            raise HTTPException(
                status_code=400, 
                detail=f"Face not detected in image: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation for SID {sid}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during embedding generation: {str(e)}"
        )