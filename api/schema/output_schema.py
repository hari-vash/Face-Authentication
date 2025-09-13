# schema/output_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional
import time

class VerificationResponse(BaseModel):
    """
    Response model for face verification.
    """
    sid: str = Field(..., description="Student ID or User ID")
    face_matched: bool = Field(..., description="Whether the face matched or not")
    confidence_score: Optional[float] = Field(None, description="Confidence score of the match (0-1)")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    message: str = Field(..., description="Human readable message about the verification result")
    
    class Config:
        schema_extra = {
            "example": {
                "sid": "STU001",
                "face_matched": True,
                "confidence_score": 0.95,
                "timestamp": "2024-01-15 10:30:45",
                "message": "Face verification successful"
            }
        }

class EmbeddingResponse(BaseModel):
    """
    Response model for embedding generation.
    """
    sid: str = Field(..., description="Student ID or User ID")
    embedding: List[float] = Field(..., description="Generated face embedding")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    message: str = Field(..., description="Status message")
    
    class Config:
        schema_extra = {
            "example": {
                "sid": "STU001",
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "timestamp": "2024-01-15 10:30:45",
                "message": "Embedding generated successfully"
            }
        }

class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(..., description="Error message")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: str = Field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    
    class Config:
        schema_extra = {
            "example": {
                "error": "Face not detected in the image",
                "status_code": 400,
                "timestamp": "2024-01-15 10:30:45"
            }
        }