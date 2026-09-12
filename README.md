# Diabetes Prediction API

FastAPI + Docker deployment of a diabetes prediction model.

## Project Structure

```
diabetes-api/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── model.py
│   └── schemas.py
├── model/
│   ├── model.pth
│   └── scaler.pkl
├── requirements.txt
├── Dockerfile
└── .dockerignore
```

## Run with Docker

```bash
docker build -t diabetes-api .
docker run -p 8000:8000 diabetes-api
```

Then open: `http://localhost:8000/docs`

## API

**POST** `/predict`

```json
{
  "gender": "Male",
  "age": 45,
  "hypertension": 1,
  "heart_disease": 0,
  "smoking_history": "former",
  "bmi": 28.5,
  "HbA1c_level": 6.8,
  "blood_glucose_level": 160
}
```

**Response**

```json
{
  "diabetes_probability": 0.97,
  "diabetes_prediction": 1
}
```

## Tech Stack

- FastAPI
- PyTorch
- scikit-learn (scaler)
- Docker
