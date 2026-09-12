from pydantic import BaseModel

class PatientInput(BaseModel):
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    smoking_history: str
    bmi: float
    HbA1c_level: float
    blood_glucose_level: int

class PredictionOutput(BaseModel):
    diabetes_probability: float
    diabetes_prediction: int
