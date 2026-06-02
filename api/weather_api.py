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


base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

static_path = os.path.join(project_root, "static")
templates_path = os.path.join(project_root, "templates")
model_path = os.path.join(project_root, "models", "model.joblib")


app = FastAPI()

app.mount(
    "/static",
    StaticFiles(directory=static_path),
    name="static"
)

templates = Jinja2Templates(directory=templates_path)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


try:
    result = joblib.load(model_path)
    print("\nConnected model!\n")

except FileNotFoundError:
    print(f"error: not found model.joblib: {model_path}")
    sys.exit(1)

except Exception as e:
    print(f"error: failed to load model: {e}")
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

    try:
        prediction = result.predict(df)

        prob = float(prediction.iloc[0])

        return {
            "rain_probability": round(prob * 100, 1)
        }

    except Exception as e:
        return {
            "error": str(e)
        }