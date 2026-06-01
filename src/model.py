import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# データの結合
df1 = pd.read_csv("data/weather1996~2006.csv")
df2 = pd.read_csv("data/weather2006~2016.csv")
df3 = pd.read_csv("data/weather2016~2026.csv")
df_weather = pd.concat([df1, df2, df3], ignore_index=True)
# 前処理
df_weather["Month"] = df_weather["Month"].ffill()
df_weather["MinTemp"] = df_weather["MinTemp"].ffill()
df_weather["AvgTemp"] = df_weather["AvgTemp"].ffill()
df_weather["AvgWindSpeed"] = df_weather["AvgWindSpeed"].ffill()
df_weather["AvgSeaLevelPressure"] = df_weather["AvgSeaLevelPressure"].ffill()
df_weather["AvgCloud"] = df_weather["AvgCloud"].fillna(df_weather["AvgCloud"].mean())
df_weather["AvgSeaLevelPressure"] = df_weather["AvgSeaLevelPressure"].fillna(df_weather["AvgSeaLevelPressure"].mean())
df_weather = df_weather.dropna(subset=["Day", "TotalPrecip", "SolarHours"])
df_weather[["Day", "Month"]] = df_weather[["Day", "Month"]].astype(int)

# 目的変数
df_weather["next_day_weather"] = (df_weather["TotalPrecip"] > 0).astype(int).shift(-1)

df = df_weather.drop(df_weather.index[-1]).copy()
df["next_day_weather"] = df["next_day_weather"].astype(int)



# 特徴量
element_name = ["AvgTemp", "TotalPrecip", "SolarHours","AvgCloud", "AvgSeaLevelPressure","AvgWindSpeed", "MinTemp"]
X = df[element_name]
y = df["next_day_weather"]

# 定数項追加
X_const = sm.add_constant(X)


# 時系列分割（過学習チェック）
X_train, X_test, y_train, y_test = train_test_split(X_const, y,test_size=0.2,random_state=42)



# モデル
model = sm.Logit(y_train, X_train)
result = model.fit()
joblib.dump(result, "models/model.joblib")