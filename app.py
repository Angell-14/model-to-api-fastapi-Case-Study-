from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="ML Inference API")

model = joblib.load("breast_cancer_model.joblib")

class PredictionInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input_data: PredictionInput):
    features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
