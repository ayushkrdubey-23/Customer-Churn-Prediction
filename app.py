from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load model, scaler, and feature names
model = joblib.load("models/churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")
feature_names = joblib.load("models/feature_names.pkl")

@app.get("/")
def home():
    return {"message": "Churn API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input dict to ordered list
        input_data = [data.get(feature, 0) for feature in feature_names]

        input_array = np.array(input_data).reshape(1, -1)

        # Scale
        input_scaled = scaler.transform(input_array)

        # Predict
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0][1]

        return {
            "prediction": int(pred),
            "churn_probability": float(prob)
        }

    except Exception as e:
        return {"error": str(e)}