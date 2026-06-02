import os
import sys
import joblib
import pandas as pd

from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI()

# staticフォルダを読み込む
app.mount("/static", StaticFiles(directory="static"), name="static")

# templatesフォルダを読み込む
templates = Jinja2Templates(directory="templates")


# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

model_path = os.path.join(project_root, "models", "model.joblib")

try:
    result = joblib.load(model_path)
    print("\nConnected model!\n")

except FileNotFoundError:
    print(f"error:not found model.joblib: {model_path}")
    sys.exit(1)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "weather_predict.html",
        {"request": request}
    )


class WeatherData(BaseModel):
    AvgTemp: float
    TotalPrecip: float
    SolarHours: float
    AvgCloud: float
    vapor_pressure: float
    AvgWindSpeed: float
    MinTemp: float


@app.post("/predict")
def predict(data: WeatherData):

    df = pd.DataFrame([{
        "AvgTemp": data.AvgTemp,
        "TotalPrecip": data.TotalPrecip,
        "SolarHours": data.SolarHours,
        "AvgCloud": data.AvgCloud,
        "vapor_pressure": data.vapor_pressure,
        "AvgWindSpeed": data.AvgWindSpeed,
        "MinTemp": data.MinTemp
    }])

    df.insert(0, "const", 1.0)

    prediction = result.predict(df)

    return {
        "rain_probability": round(float(prediction.iloc[0] * 100), 1)
    }