import os
import sys
import joblib
import pandas as pd

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

static_path = os.path.join(project_root, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")


@app.get("/")
def index():
    html_path = os.path.join(project_root, "templates", "weather_predict.html")
    return FileResponse(html_path)


model_path = os.path.join(project_root, "models", "model.joblib")

try:
    result = joblib.load(model_path)
    print("\nConnected model!\n")
except FileNotFoundError:
    print(f"error: not found model.joblib: {model_path}")
    sys.exit(1)


class WeatherData(BaseModel):
    Month: float
    MinTemp: float
    AvgTemp: float
    TotalPrecip: float
    vapor_pressure: float
    AvgCloud: float


@app.post("/predict")
def predict(data: WeatherData):

    df = pd.DataFrame([{
        "Month": data.Month,
        "MinTemp": data.MinTemp,
        "AvgTemp": data.AvgTemp,
        "TotalPrecip": data.TotalPrecip,
        "vapor_pressure": data.vapor_pressure,
        "AvgCloud": data.AvgCloud
    }])

    df.insert(0, "const", 1.0)

    prediction = result.predict(df)

    rain_probability = round(float(prediction.iloc[0] * 100), 1)

    return {
        "rain_probability": rain_probability
    }