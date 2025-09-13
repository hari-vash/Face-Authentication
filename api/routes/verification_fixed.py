# # routes/verification.py
# import numpy as np
# import cv2
# import io
# from PIL import Image
# from fastapi import APIRouter, HTTPException, UploadFile, File, Form
# from deepface import DeepFace
# from typing import List
# import json

# from schema.output_schema import VerificationResponse, ErrorResponse
# from logger.logger import logger

# router = APIRouter(prefix="/verification", tags=["Face Verification"])

# def preprocess_image(image_data: bytes) -> np.ndarray:
#     """
#     Preprocess uploaded image for DeepFace processing.
#     """
#     try:
#         # Convert bytes to PIL Image
#         image = Image.open(io.BytesIO(image_data))
        
#         # Convert to RGB if necessary
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array
#         img_array = np.array(image)
        
#         return img_array
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
#     """
#     Calculate cosine similarity between two embeddings.
#     """
#     try:
#         # Convert to numpy arrays
#         emb1 = np.array(embedding1)
#         emb2 = np.array(embedding2)
        
#         # Calculate cosine similarity
#         dot_product = np.dot(emb1, emb2)
#         norm1 = np.linalg.norm(emb1)
#         norm2 = np.linalg.norm(emb2)
        
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
            
#         similarity = dot_product / (norm1 * norm2)
#         return float(similarity)
#     except Exception as e:
#         logger.error(f"Error calculating similarity: {str(e)}")
#         return 0.0

# @router.post("/verify", response_model=VerificationResponse)
# async def verify_face(
#     sid: str = Form(..., description="Student ID or User ID"),
#     stored_embedding: str = Form(..., description="JSON string of stored face embedding"),
#     image: UploadFile = File(..., description="Camera image for verification")
# ):
#     """
#     Verify face by comparing stored embedding with camera image.
    
#     - **sid**: Student ID or User ID
#     - **stored_embedding**: Previously stored face embedding (as JSON string)
#     - **image**: Camera image file for verification
#     """
#     try:
#         logger.info(f"Starting face verification for SID: {sid}")
        
#         # Parse stored embedding from JSON string
#         try:
#             stored_emb = json.loads(stored_embedding)
#             if not isinstance(stored_emb, list):
#                 raise ValueError("Stored embedding must be a list of numbers")
#         except (json.JSONDecodeError, ValueError) as e:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid stored_embedding format: {str(e)}"
#             )
        
#         # Validate image file
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read and preprocess image
#         image_data = await image.read()
#         img_array = preprocess_image(image_data)
        
#         # Generate embedding for the uploaded image using DeepFace
#         try:
#             # Generate embedding using DeepFace
#             current_embedding = DeepFace.represent(
#                 img_path=img_array,
#                 model_name='Facenet',  # You can change to 'VGG-Face', 'ArcFace', etc.
#                 enforce_detection=True
#             )[0]["embedding"]
            
#             logger.info(f"Successfully generated embedding for SID: {sid}")
            
#         except Exception as e:
#             logger.error(f"Failed to generate embedding for SID {sid}: {str(e)}")
#             return VerificationResponse(
#                 sid=sid,
#                 face_matched=False,
#                 confidence_score=0.0,
#                 message=f"Face not detected in image: {str(e)}"
#             )
        
#         # Calculate similarity between stored and current embeddings
#         similarity_score = calculate_similarity(stored_emb, current_embedding)
        
#         # Set threshold for face matching (you can adjust this)
#         threshold = 0.6
#         face_matched = similarity_score >= threshold
        
#         message = (
#             f"Face verification successful - Similarity: {similarity_score:.3f}" 
#             if face_matched 
#             else f"Face verification failed - Similarity: {similarity_score:.3f} (threshold: {threshold})"
#         )
        
#         logger.info(f"Face verification completed for SID {sid}: {face_matched}")
        
#         return VerificationResponse(
#             sid=sid,
#             face_matched=face_matched,
#             confidence_score=similarity_score,
#             message=message
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during verification for SID {sid}: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Internal server error during verification: {str(e)}"
#         )

# #######################################################################################################################################################################################################
# routes/verification.py
# import numpy as np
# import cv2
# import io
# from PIL import Image
# from fastapi import APIRouter, HTTPException, UploadFile, File, Form
# from deepface import DeepFace
# from typing import List
# import json

# from schema.output_schema import VerificationResponse, ErrorResponse
# from logger.logger import logger

# router = APIRouter(prefix="/verification", tags=["Face Verification"])

# def preprocess_image(image_data: bytes) -> np.ndarray:
#     """
#     Preprocess uploaded image for DeepFace processing.
#     """
#     try:
#         # Convert bytes to PIL Image
#         image = Image.open(io.BytesIO(image_data))
        
#         # Convert to RGB if necessary
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array
#         img_array = np.array(image)
        
#         return img_array
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
#     """
#     Calculate cosine similarity between two embeddings.
#     """
#     try:
#         # Convert to numpy arrays
#         emb1 = np.array(embedding1)
#         emb2 = np.array(embedding2)
        
#         # Calculate cosine similarity
#         dot_product = np.dot(emb1, emb2)
#         norm1 = np.linalg.norm(emb1)
#         norm2 = np.linalg.norm(emb2)
        
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
            
#         similarity = dot_product / (norm1 * norm2)
#         return float(similarity)
#     except Exception as e:
#         logger.error(f"Error calculating similarity: {str(e)}")
#         return 0.0

# @router.post("/verify", response_model=VerificationResponse)
# async def verify_face(
#     sid: str = Form(..., description="Student ID or User ID"),
#     stored_embedding: str = Form(..., description="JSON string of stored face embedding"),
#     image: UploadFile = File(..., description="Camera image for verification")
# ):
#     """
#     Verify face by comparing stored embedding with camera image.
    
#     - **sid**: Student ID or User ID
#     - **stored_embedding**: Previously stored face embedding (JSON array [1,2,3] or comma-separated 1,2,3)
#     - **image**: Camera image file for verification
#     """
#     try:
#         logger.info(f"Starting face verification for SID: {sid}")
        
#         # Parse stored embedding from JSON string
#         try:
#             stored_emb = json.loads(stored_embedding)
#             if not isinstance(stored_emb, list):
#                 raise ValueError("Stored embedding must be a list of numbers")
#         except (json.JSONDecodeError, ValueError) as e:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid stored_embedding format: {str(e)}"
#             )
        
#         # Validate image file
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read and preprocess image
#         image_data = await image.read()
#         img_array = preprocess_image(image_data)
        
#         # Generate embedding for the uploaded image using DeepFace
#         try:
#             # Generate embedding using DeepFace
#             current_embedding = DeepFace.represent(
#                 img_path=img_array,
#                 model_name='Facenet',  # You can change to 'VGG-Face', 'ArcFace', etc.
#                 enforce_detection=True
#             )[0]["embedding"]
            
#             logger.info(f"Successfully generated embedding for SID: {sid}")
            
#         except Exception as e:
#             logger.error(f"Failed to generate embedding for SID {sid}: {str(e)}")
#             return VerificationResponse(
#                 sid=sid,
#                 face_matched=False,
#                 confidence_score=0.0,
#                 message=f"Face not detected in image: {str(e)}"
#             )
        
#         # Calculate similarity between stored and current embeddings
#         similarity_score = calculate_similarity(stored_emb, current_embedding)
        
#         # Set threshold for face matching (you can adjust this)
#         threshold = 0.6
#         face_matched = similarity_score >= threshold
        
#         message = (
#             f"Face verification successful - Similarity: {similarity_score:.3f}" 
#             if face_matched 
#             else f"Face verification failed - Similarity: {similarity_score:.3f} (threshold: {threshold})"
#         )
        
#         logger.info(f"Face verification completed for SID {sid}: {face_matched}")
        
#         return VerificationResponse(
#             sid=sid,
#             face_matched=face_matched,
#             confidence_score=similarity_score,
#             message=message
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during verification for SID {sid}: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Internal server error during verification: {str(e)}"
#         )

###################################################################################################################################################################################################33

# # routes/verification.py
# import numpy as np
# import cv2
# import io
# from PIL import Image
# from fastapi import APIRouter, HTTPException, UploadFile, File, Form
# from deepface import DeepFace
# from typing import List
# import json

# from schema.output_schema import VerificationResponse, ErrorResponse
# from logger.logger import logger

# router = APIRouter(prefix="/verification", tags=["Face Verification"])

# def preprocess_image(image_data: bytes) -> np.ndarray:
#     """
#     Preprocess uploaded image for DeepFace processing.
#     """
#     try:
#         # Convert bytes to PIL Image
#         image = Image.open(io.BytesIO(image_data))
        
#         # Convert to RGB if necessary
#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        
#         # Convert to numpy array
#         img_array = np.array(image)
        
#         return img_array
#     except Exception as e:
#         logger.error(f"Error preprocessing image: {str(e)}")
#         raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

# def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
#     """
#     Calculate cosine similarity between two embeddings.
#     """
#     try:
#         # Convert to numpy arrays
#         emb1 = np.array(embedding1)
#         emb2 = np.array(embedding2)
        
#         # Calculate cosine similarity
#         dot_product = np.dot(emb1, emb2)
#         norm1 = np.linalg.norm(emb1)
#         norm2 = np.linalg.norm(emb2)
        
#         if norm1 == 0 or norm2 == 0:
#             return 0.0
            
#         similarity = dot_product / (norm1 * norm2)
#         return float(similarity)
#     except Exception as e:
#         logger.error(f"Error calculating similarity: {str(e)}")
#         return 0.0

# @router.post("/test-upload", response_model=dict)
# async def test_file_upload(
#     test_file: UploadFile = File(..., description="Test image upload")
# ):
#     """
#     Simple test endpoint to verify file uploads are working correctly.
#     """
#     try:
#         content = await test_file.read()
#         return {
#             "filename": test_file.filename,
#             "content_type": test_file.content_type,
#             "file_size": len(content),
#             "status": "success"
#         }
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

# @router.post("/verify", response_model=VerificationResponse)
# async def verify_face(
#     sid: str = Form(..., description="Student ID or User ID"),
#     stored_embedding: str = Form(..., description="JSON string of stored face embedding"),
#     image: UploadFile = File(..., description="Camera image for verification")
# ):
#     """
#     Verify face by comparing stored embedding with camera image.
    
#     - **sid**: Student ID or User ID
#     - **stored_embedding**: Previously stored face embedding (JSON array [1,2,3] or comma-separated 1,2,3)
#     - **image**: Camera image file for verification
#     """
#     try:
#         logger.info(f"Starting face verification for SID: {sid}")
        
#         # Parse stored embedding from JSON string
#         try:
#             stored_emb = json.loads(stored_embedding)
#             if not isinstance(stored_emb, list):
#                 raise ValueError("Stored embedding must be a list of numbers")
#         except (json.JSONDecodeError, ValueError) as e:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid stored_embedding format: {str(e)}"
#             )
        
#         # Validate image file
#         if not image.content_type.startswith('image/'):
#             raise HTTPException(status_code=400, detail="File must be an image")
        
#         # Read and preprocess image
#         image_data = await image.read()
#         img_array = preprocess_image(image_data)
        
#         # Generate embedding for the uploaded image using DeepFace
#         try:
#             # Generate embedding using DeepFace
#             current_embedding = DeepFace.represent(
#                 img_path=img_array,
#                 model_name='Facenet',  # You can change to 'VGG-Face', 'ArcFace', etc.
#                 enforce_detection=True
#             )[0]["embedding"]
            
#             logger.info(f"Successfully generated embedding for SID: {sid}")
            
#         except Exception as e:
#             logger.error(f"Failed to generate embedding for SID {sid}: {str(e)}")
#             return VerificationResponse(
#                 sid=sid,
#                 face_matched=False,
#                 confidence_score=0.0,
#                 message=f"Face not detected in image: {str(e)}"
#             )
        
#         # Calculate similarity between stored and current embeddings
#         similarity_score = calculate_similarity(stored_emb, current_embedding)
        
#         # Set threshold for face matching (you can adjust this)
#         threshold = 0.6
#         face_matched = similarity_score >= threshold
        
#         message = (
#             f"Face verification successful - Similarity: {similarity_score:.3f}" 
#             if face_matched 
#             else f"Face verification failed - Similarity: {similarity_score:.3f} (threshold: {threshold})"
#         )
        
#         logger.info(f"Face verification completed for SID {sid}: {face_matched}")
        
#         return VerificationResponse(
#             sid=sid,
#             face_matched=face_matched,
#             confidence_score=similarity_score,
#             message=message
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error during verification for SID {sid}: {str(e)}")
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Internal server error during verification: {str(e)}"
#         )

#########################################################################################################################################################################################################################
# routes/verification_fixed.py
import numpy as np
import io
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from deepface import DeepFace
from typing import List
import json

from schema.output_schema import VerificationResponse, ErrorResponse
from logger.logger import logger

router = APIRouter(prefix="/verification", tags=["Face Verification"])

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

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    """
    try:
        # Convert to numpy arrays
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

# Simple test endpoint
@router.post("/test-upload")
async def test_file_upload(file: UploadFile = File(...)):
    """Test file upload"""
    try:
        contents = await file.read()
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(contents),
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Working verification endpoint
@router.post("/verify")
async def verify_face(
    request: Request,
    sid: str = Form(),
    stored_embedding: str = Form(),
    image: UploadFile = File()
):
    """
    Verify face by comparing stored embedding with camera image.
    """
    try:
        logger.info(f"Starting face verification for SID: {sid}")
        logger.info(f"Received file: {image.filename}, Content-Type: {image.content_type}")
        
        # Parse stored embedding - handle both JSON array and comma-separated values
        try:
            # First try to parse as JSON array
            if stored_embedding.strip().startswith('[') and stored_embedding.strip().endswith(']'):
                stored_emb = json.loads(stored_embedding)
            else:
                # Handle comma-separated values
                stored_emb = [float(x.strip()) for x in stored_embedding.split(',')]
            
            if not isinstance(stored_emb, list) or len(stored_emb) == 0:
                raise ValueError("Stored embedding must be a non-empty list of numbers")
                
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid stored_embedding format. Please provide either a JSON array [1,2,3] or comma-separated values 1,2,3. Error: {str(e)}"
            )
        
        # Validate image file
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
            
        if not image.content_type or not image.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400, 
                detail=f"File must be an image. Received content type: {image.content_type}"
            )
        
        # Read and preprocess image
        image_data = await image.read()
        img_array = preprocess_image(image_data)
        
        logger.info(f"Image preprocessed successfully. Shape: {img_array.shape}")
        
        # Generate embedding for the uploaded image using DeepFace
        try:
            # Generate embedding using DeepFace
            current_embedding = DeepFace.represent(
                img_path=img_array,
                model_name='VGG-Face',  # You can change to 'VGG-Face', 'ArcFace', etc.
                enforce_detection=True
            )[0]["embedding"]
            
            logger.info(f"Successfully generated embedding for SID: {sid}")
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for SID {sid}: {str(e)}")
            return JSONResponse(
                status_code=400,
                content={
                    "sid": sid,
                    "face_matched": False,
                    "confidence_score": 0.0,
                    "message": f"Face not detected in image: {str(e)}",
                    "timestamp": "2024-01-15 10:30:45"
                }
            )
        
        # Calculate similarity between stored and current embeddings
        similarity_score = calculate_similarity(stored_emb, current_embedding)
        
        # Set threshold for face matching (you can adjust this)
        threshold = 0.6
        face_matched = similarity_score >= threshold
        
        message = (
            f"Face verification successful - Similarity: {similarity_score:.3f}" 
            if face_matched 
            else f"Face verification failed - Similarity: {similarity_score:.3f} (threshold: {threshold})"
        )
        
        logger.info(f"Face verification completed for SID {sid}: {face_matched}")
        
        return JSONResponse(content={
            "sid": sid,
            "face_matched": face_matched,
            "confidence_score": similarity_score,
            "message": message,
            "timestamp": "2024-01-15 10:30:45"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during verification for SID {sid}: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during verification: {str(e)}"
        )