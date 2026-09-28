from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi import Depends
from sqlalchemy.orm import Session

from src.predict import predict_image
from api.schemas import PredictionResponse, PredictionHistory
from database.database import get_db
from database.models import Prediction

import shutil
import os
import uuid
import traceback

router = APIRouter(
    prefix="/predict",
    tags=["Prediction"]
)


@router.post("/", response_model=PredictionResponse)
def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/jpg"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Only JPG and PNG images are allowed"
        )

    file_extension = os.path.splitext(file.filename)[1]

    file_name = f"{uuid.uuid4()}{file_extension}"

    file_path = os.path.join("uploads", file_name)

    try:

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        predicted_class, confidence, probabilities = predict_image(file_path)

        prediction_record = Prediction(
            prediction=predicted_class,
            confidence=confidence
        )

        db.add(prediction_record)
        db.commit()

        return {
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": probabilities
        }

    except Exception as e:

        db.rollback()
        print("PREDICTION ERROR:", repr(e))
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if os.path.exists(file_path):
            os.remove(file_path)


@router.get("/history", response_model=list[PredictionHistory])
def get_prediction_history(
    db: Session = Depends(get_db)
):

    predictions = db.query(Prediction).all()

    return predictions
