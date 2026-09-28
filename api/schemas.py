from pydantic import BaseModel
from datetime import datetime


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


class PredictionHistory(BaseModel):
    id: int
    prediction: str
    confidence: float
    created_at: datetime

    class Config:
        from_attributes = True