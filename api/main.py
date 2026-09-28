from fastapi import FastAPI

from api.routers import prediction, health

from database.database import engine
from database import models


models.Base.metadata.create_all(bind=engine)


app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for brain tumor MRI image classification",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Brain Tumor Detection API is running"
    }


app.include_router(prediction.router)
app.include_router(health.router)