# schema/input_schema.py
from pydantic import BaseModel, Field
from typing import List, Optional
from fastapi import UploadFile, File, Form

class VerificationRequest(BaseModel):
    """
    Model for face verification request.
    Contains SID, stored embedding, and camera image for comparison.
    """
    sid: str = Field(..., description="Student ID or User ID", example="STU001")
    stored_embedding: List[float] = Field(
        ..., 
        description="Previously stored face embedding for comparison",
        example=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    
    class Config:
        schema_extra = {
            "example": {
                "sid": "STU001",
                "stored_embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
        }

class EmbeddingGenerationRequest(BaseModel):
    """
    Model for embedding generation request.
    Contains SID for identification.
    """
    sid: str = Field(..., description="Student ID or User ID", example="STU001")
    
    class Config:
        schema_extra = {
            "example": {
                "sid": "STU001"
            }
        }