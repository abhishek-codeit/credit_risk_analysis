from fastapi import FastAPI
from pathlib import Path
import xgboost as xgb
import pandas as pd
app = FastAPI()

model = xgb.XGBClassifier()
model.load_model(Path(__file__).resolve().parent / 'credit_score_mlops' / 'model' / 'model.ubj')

@app.post("/predict")
def prediction(data :dict):

    input = pd.DataFrame([data])

    prediction = model.predict(input)
    prediction_prob = model.predict_proba(input)[:,-1]

    return {
        "fraud_prediction": int(prediction[0]),
        "fraud_probability": float(prediction_prob[0])
    }
    