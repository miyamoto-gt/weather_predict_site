# 天気予測Webアプリ

## 概要

本プロジェクトは、前回開発した「翌日の降雨予測モデル」においてのランダム分割（weather_random_state.py）のモデルをWebアプリケーションとして利用できるようにすることを目的として取り組んだものである。
前回の機械学習プロジェクトでは、気象庁の約30年分の気象データを用いてロジスティック回帰による翌日の降雨予測モデルを構築し、約72％の予測精度を達成した。
しかし、学習済みモデルを作成しただけでは一般ユーザーが利用することはできない。
そこで今回は、学習済みモデルをAPI化し、Webブラウザから気象データを入力することで降雨確率を予測できるシステムの開発を行った。
-------------------------------------------------------------
### ※前回のモデル構築URL
https://github.com/miyamoto-gt/weather.git

### ランダム分割を選択した理由
時系列分割モデルは高いAccuracyを示したが、実際には全てのデータを「雨」と予測する多数派予測に近い状態であった。そのため、本プロジェクトでは実用上の予測能力を重視し、「雨あり・雨なし」を区別できるランダム分割モデルを採用した。


## ディレクトリ構造

weather_predict_site/
├── api/
│   └── weather_api.py
│
├── data/
│   ├── weather1996~2006.csv
│   ├── weather2006~2016.csv
│   └── weather2016~2026.csv
│
├── models/
│   └── model.joblib
│
├── src/
│   └── model.py
│
└── README.md


## 使用技術

### バックエンド

- Python
- FastAPI
- Joblib

### データ分析・機械学習

- Pandas
- Statsmodels
- Logistic Regression

---

## システム構成

気象データ
↓
model.py
↓
学習済みモデル（model.joblib）
↓
weather_api.py
↓
FastAPI
↓
ユーザー　　　ここ追加

学習済みモデルをAPI経由で利用できる構成とした。

---

## 使用データ

気象庁の過去データを使用

https://www.data.jma.go.jp/risk/obsdl/index.php

対象地域：石川県金沢市

使用期間

- 1996年〜2026年

使用特徴量

- 平均気温（AvgTemp）
- 最低気温（MinTemp）
- 降水量（TotalPrecip）
- 日照時間（SolarHours）
- 雲量（AvgCloud）
- 平均風速（AvgWindSpeed）
- 海面気圧（AvgSeaLevelPressure）

---

## モデル構築

前回開発したロジスティック回帰モデルを利用した。

```python
model = sm.Logit(y_train, X_train)
result = model.fit()