import os ,sys 
import joblib
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware# モデルの読み込み
app =FastAPI()
# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

base_dir=os.path.dirname(os.path.abspath(__file__))
project_root=os.path.dirname(base_dir)

model_path=os.path.join(project_root,"models","model.joblib")

try:
    result=joblib.load(model_path)
    print("\nConnected model!\n")

except FileNotFoundError:
    print(f"error:not found model.joblib:{model_path}")
    sys.exit(1)

class weatherdata(BaseModel):
    AvgTemp:float
    TotalPrecip:float
    SolarHours:float
    AvgCloud:float
    vapor_pressure:float
    AvgWindSpeed:float
    MinTemp:float
@app.post("/predict")
def predict(data: weatherdata):

    df=pd.DataFrame([{
        "AvgTemp":data.AvgTemp,
        "TotalPrecip":data.TotalPrecip,
        "SolarHours":data.SolarHours,
        "AvgCloud":data.AvgCloud,
        "vapor_pressure":data.vapor_pressure,
        "AvgWindSpeed":data.AvgWindSpeed,
        "MinTemp":data.MinTemp
    }])

    df.insert(0,"const",1.0)
    prediction=result.predict(df)
    return{
        "rain_probability":round(float(prediction.iloc[0]*100),1)
    }