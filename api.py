from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi import FastAPI
import joblib

model = joblib.load("house_price_model.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://house-price-frontend-ruddy.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "API is running"}


@app.post("/predict")
def predict(sqft: int,bedrooms: int, bathrooms: int, age: int):

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