import pandas as pd
from fastapi import FastAPI
import joblib

model = joblib.load("house_price_model.pkl")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "API is running"}


@app.get("/predict")
def predict(bedrooms: int, bathrooms: int, sqft: int, age: int):

    data = pd.DataFrame({
        "Square_Feet": [sqft],
        "Bedrooms": [bedrooms],
        "Bathrooms": [bathrooms],
        "Age_Years": [age]
    })

    prediction = model.predict(data)

    return {
        "Square_Feet": sqft,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_Years": age,
        "prediction": float(prediction[0])
    }


#http://127.0.0.1:8000/predict?bedrooms=3&bathrooms=2&sqft=1500&age=10