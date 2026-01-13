from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="API Iris Classification")

# load model
# to run the local server # python -m uvicorn main:app --reload
# if not run. pip install fastapi uvicorn joblib numpy scikit-learn
# postman to test http://127.0.0.1:8000/predict

model = joblib.load('iris_model.joblib')

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def home():
    return {"mensagem": "API Iris pronta! Envia POST para /predict"}

@app.post("/predict")
def predict(features: IrisFeatures):
    data = np.array([[features.sepal_length,
                      features.sepal_width,
                      features.petal_length,
                      features.petal_width]])
    
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0].max()  # confiança
    
    return {
        "especie_prevista": prediction,
        "probabilidade": round(probability, 4)
    }

#POSTMAN FLOWER TEST EXAMPLE RAW JSON - EXPECTED RESULT = IRIS-Virginica
'''{ 
    "sepal_length": 5.9,
    "sepal_width": 3.0,
    "petal_length": 5.1,
    "petal_width": 1.8 
}'''