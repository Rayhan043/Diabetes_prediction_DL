import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"

FEATURE_ORDER = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level",
    "gender_Male",
    "gender_Other",
    "smoking_history_current",
    "smoking_history_ever",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_not current",
]


class MyMLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.output_layer(x))
        return x


def load_model_and_scaler():
    if "MyMLP" not in sys.modules["__main__"].__dict__:
        sys.modules["__main__"].MyMLP = MyMLP

    with MODEL_PATH.open("rb") as model_file:
        loaded_model = pickle.load(model_file)

    if isinstance(loaded_model, MyMLP):
        model = loaded_model
    elif isinstance(loaded_model, dict):
        model = MyMLP(input_size=len(FEATURE_ORDER))
        model.load_state_dict(loaded_model)
    else:
        raise TypeError(
            f"Unsupported model artifact in {MODEL_PATH}: {type(loaded_model).__name__}"
        )

    model.eval()
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    return model, scaler


def preprocess(data: dict) -> np.ndarray:
    gender = str(data.get("gender", "")).strip().lower()
    smoking_history = str(data.get("smoking_history", "")).strip().lower()

    row = {
        "age": float(data.get("age", 0)),
        "hypertension": int(data.get("hypertension", 0)),
        "heart_disease": int(data.get("heart_disease", 0)),
        "bmi": float(data.get("bmi", 0)),
        "HbA1c_level": float(data.get("HbA1c_level", 0)),
        "blood_glucose_level": float(data.get("blood_glucose_level", 0)),
        "gender_Male": 1.0 if gender == "male" else 0.0,
        "gender_Other": 1.0 if gender == "other" else 0.0,
        "smoking_history_current": 1.0 if smoking_history == "current" else 0.0,
        "smoking_history_ever": 1.0 if smoking_history == "ever" else 0.0,
        "smoking_history_former": 1.0 if smoking_history == "former" else 0.0,
        "smoking_history_never": 1.0 if smoking_history == "never" else 0.0,
        "smoking_history_not current": 1.0 if smoking_history == "not current" else 0.0,
    }
    return np.array([[row[col] for col in FEATURE_ORDER]], dtype=np.float32)