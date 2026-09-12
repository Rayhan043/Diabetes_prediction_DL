import torch
from fastapi import FastAPI

from app.model import load_model_and_scaler, preprocess
from app.schemas import PatientInput, PredictionOutput

app = FastAPI(title="Diabetes Prediction API")

model, scaler = load_model_and_scaler()


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(patient: PatientInput):
    x = preprocess(patient.model_dump())

    if scaler is not None:
        x = scaler.transform(x)

    x_tensor = torch.tensor(x, dtype=torch.float32)

    with torch.no_grad():
        prob = model(x_tensor).squeeze().item()

    return PredictionOutput(
        diabetes_probability=round(float(prob), 4),
        diabetes_prediction=1 if float(prob) >= 0.5 else 0,
    )
