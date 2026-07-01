import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import joblib

# データ取得
df_weather = pd.read_csv("data/weather2011~2016.csv")

# 前処理
df_weather["MinTemp"] = df_weather["MinTemp"].ffill()
df_weather["AvgTemp"] = df_weather["AvgTemp"].ffill()
df_weather["TotalPrecip"] = df_weather["TotalPrecip"].fillna(0)
df_weather["SolarHours"] = df_weather["SolarHours"].ffill()
df_weather["AvgWindSpeed"] = df_weather["AvgWindSpeed"].ffill()
df_weather["vapor_pressure"] = df_weather["vapor_pressure"].ffill()
df_weather= df_weather.dropna(subset="AvgCloud")

# 目的変数
df_weather["next_day_weather"] = (df_weather["TotalPrecip"] > 0).astype(int).shift(-1)
df = df_weather.drop(df_weather.index[-1]).copy()
df["next_day_weather"] = df["next_day_weather"].astype(int)

element_name = ["Month","MinTemp","AvgTemp", "TotalPrecip", "vapor_pressure","AvgCloud"]
explanatory_variables = df[element_name]
objective_variable = df["next_day_weather"]
X_all_const = sm.add_constant(explanatory_variables)

X_train, X_test, y_train, y_test = train_test_split(
    X_all_const, objective_variable, test_size=0.2, shuffle=False
)




# モデル
model = sm.Logit(y_train, X_train)
result = model.fit()
joblib.dump(result, "models/model.joblib")
