# 天気予測Webアプリ

## 概要

本プロジェクトは、前回開発した「翌日の降雨予測モデル」をWebアプリケーションとして利用できるようにすることを目的として取り組んだものである。

前回の機械学習プロジェクトでは、気象庁の約30年分の気象データを用いて、ロジスティック回帰による翌日の降雨予測モデルを構築した。
その結果、約72％の予測精度を確認できた。

しかし、学習済みモデルを作成しただけでは、Python環境を持たない一般ユーザーが利用することはできない。
そこで今回は、学習済みモデルを `joblib` で保存し、FastAPIを用いてAPI化した。さらに、HTML・CSS・JavaScriptで作成したフロントエンドと連携することで、Webブラウザから気象データを入力し、翌日の降雨確率を確認できるWebアプリとして実装した。

---

## 公開Webアプリ

https://weather-predict-site.onrender.com

---

## 前回のモデル構築リポジトリ

https://github.com/miyamoto-gt/weather.git

---

## アプリの目的

本アプリの目的は、機械学習モデルを作成するだけでなく、実際にWebアプリケーションとして利用できる形にすることである。

具体的には、以下の流れを実装した。

```text
気象データの収集
↓
前処理
↓
ロジスティック回帰モデルの学習
↓
学習済みモデルの保存
↓
FastAPIによるAPI化
↓
JavaScriptからAPIへリクエスト
↓
Web画面に予測結果を表示
↓
Renderで公開
```

これにより、Pythonコードを直接実行しなくても、Webブラウザから気象データを入力するだけで、翌日の降雨確率を確認できるようにした。

---

## ディレクトリ構造

```text
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
├── static/
│   ├── weather_predict.css
│   └── weather_predict.js
│
├── templates/
│   └── weather_predict.html
│
├── requirements.txt
└── README.md
```

---

## 使用技術

### バックエンド

* Python
* FastAPI
* Uvicorn
* Pydantic
* Joblib

### データ分析・機械学習

* Pandas
* Statsmodels
* scikit-learn
* NumPy
* Logistic Regression

### フロントエンド

* HTML
* CSS
* JavaScript

### デプロイ

* Render

---

## 使用データ

本プロジェクトでは、気象庁の過去の気象データを使用した。

データ取得元：

https://www.data.jma.go.jp/risk/obsdl/index.php

対象地域：

* 石川県金沢市

使用期間：

* 1996年〜2026年

使用した特徴量：

* 平均気温（AvgTemp）
* 最低気温（MinTemp）
* 降水量（TotalPrecip）
* 日照時間（SolarHours）
* 雲量（AvgCloud）
* 平均蒸気圧（vapor_pressure）
* 平均風速（AvgWindSpeed）

---

## システム構成

```text
気象データCSV
↓
model.py
↓
前処理・特徴量選択・モデル学習
↓
学習済みモデル model.joblib
↓
weather_api.py
↓
FastAPIによる予測API
↓
JavaScriptから /predict にリクエスト
↓
降雨確率をJSON形式で返す
↓
Web画面に予測結果を表示
↓
ユーザー
```

本アプリでは、`model.py` で学習したロジスティック回帰モデルを `model.joblib` として保存し、`weather_api.py` で読み込んでいる。
ユーザーがWeb画面に気象データを入力すると、JavaScriptからFastAPIの `/predict` にPOSTリクエストが送信される。
API側では受け取った値をDataFrameに変換し、保存済みモデルに入力することで翌日の降雨確率を計算する。
計算結果はJSON形式でフロントエンドへ返され、Web画面上に表示される。

---

## モデル構築

### データの読み込みと結合

`model.py` では、1996年〜2026年までの気象データを3つのCSVファイルから読み込み、`pd.concat()` を用いて1つのデータフレームに結合した。

```python
df1 = pd.read_csv("data/weather1996~2006.csv")
df2 = pd.read_csv("data/weather2006~2016.csv")
df3 = pd.read_csv("data/weather2016~2026.csv")

df_weather = pd.concat([df1, df2, df3], ignore_index=True)
```

複数のCSVファイルを結合することで、より多くの過去データをモデルの学習に利用できるようにした。

---

### 前処理

気象データには欠損値が含まれていたため、特徴量ごとに補完処理を行った。

```python
df_weather["Month"] = df_weather["Month"].ffill()
df_weather["MinTemp"] = df_weather["MinTemp"].ffill()
df_weather["AvgTemp"] = df_weather["AvgTemp"].ffill()
df_weather["AvgWindSpeed"] = df_weather["AvgWindSpeed"].ffill()
df_weather["vapor_pressure"] = df_weather["vapor_pressure"].ffill()
df_weather["AvgCloud"] = df_weather["AvgCloud"].fillna(df_weather["AvgCloud"].mean())
df_weather["vapor_pressure"] = df_weather["vapor_pressure"].fillna(df_weather["vapor_pressure"].mean())
df_weather = df_weather.dropna(subset=["Day", "TotalPrecip", "SolarHours"])
```

気温、風速、蒸気圧などの連続的な値は、前後の日付で大きく急変しにくいと考え、前方補完を使用した。
一方で、雲量など一部の特徴量には平均値補完を用いた。
また、日付や降水量、日照時間など、目的変数や特徴量として重要な値が欠損している行は削除した。

---

### 目的変数の作成

本モデルでは、翌日に雨が降ったかどうかを表す `next_day_weather` を目的変数として作成した。

```python
df_weather["next_day_weather"] = (df_weather["TotalPrecip"] > 0).astype(int).shift(-1)
```

降水量が0より大きい場合を「雨あり」として `1`、降水量が0の場合を「雨なし」として `0` に変換した。
さらに、`shift(-1)` を使うことで、当日の気象データに対して翌日の天気を対応させている。

例えば、6月1日の気温や雲量などのデータに対して、6月2日に雨が降ったかどうかを目的変数として設定する。

```text
6月1日の気象データ → 6月2日に雨が降ったか
6月2日の気象データ → 6月3日に雨が降ったか
```

これにより、「今日の気象データから翌日の降雨を予測する」形にした。

---

### 特徴量

モデルに入力する特徴量は以下の7つである。

```python
element_name = [
    "AvgTemp",
    "TotalPrecip",
    "SolarHours",
    "AvgCloud",
    "vapor_pressure",
    "AvgWindSpeed",
    "MinTemp"
]
```

これらの特徴量を選んだ理由は、気温、降水量、日照時間、雲量、蒸気圧、風速などが翌日の天気に関係すると考えたためである。

---

### 定数項の追加

statsmodelsのロジスティック回帰では、切片を明示的に追加する必要があるため、`sm.add_constant()` を用いて定数項を追加した。

```python
X_const = sm.add_constant(X)
```

API側で予測を行う際にも、学習時と同じ列構成にするため、`const` を追加している。

---

### ランダム分割

学習データとテストデータは、`train_test_split()` を用いて分割した。

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_const,
    y,
    test_size=0.2,
    random_state=1
)
```

`test_size=0.2` とすることで、全体の20％をテストデータ、80％を学習データとして使用した。
また、`random_state` を指定することで、実行するたびに分割結果が変わらないようにした。

`random_state` の値を変えると、学習データとテストデータに選ばれる日が変わるため、AccuracyやPrecisionなどの評価結果も変化する。
これは、気象データには雨の日と晴れの日の偏りや季節性があるためである。

そのため、本プロジェクトでは評価結果の再現性を保つために `random_state` を固定した。

---

### ランダム分割モデルを採用した理由

前回のモデル構築では、ランダム分割と時系列分割の比較を行った。
時系列分割モデルは高いAccuracyを示したが、実際には全てのデータを「雨」と予測する多数派予測に近い状態であった。

そのため、単にAccuracyが高いモデルではなく、「雨あり」と「雨なし」を区別できるかを重視した。
本プロジェクトでは、実用上の予測能力を重視し、ランダム分割モデルを採用した。

---

### モデルの学習

モデルには、statsmodelsの `Logit` を用いたロジスティック回帰を使用した。

```python
model = sm.Logit(y_train, X_train)
result = model.fit()
```

ロジスティック回帰を用いることで、翌日に雨が降る確率を0〜1の範囲で出力できる。
API側では、この値を100倍してパーセント表示に変換している。

---

## 学習済みモデルの保存

学習済みモデルは、`joblib` を用いて `models/model.joblib` として保存した。

```python
joblib.dump(result, "models/model.joblib")
```

モデルを一度 `model.joblib` として保存する理由は、Webアプリで予測を行うたびに毎回モデルを学習し直す必要をなくすためである。

もしユーザーが予測するたびに、CSVの読み込み、欠損値処理、特徴量作成、モデル学習を毎回実行すると、処理が重くなってしまう。
そこで、モデルの学習は `model.py` で一度だけ行い、学習済みモデルをファイルとして保存した。

API側では、この保存済みモデルを読み込み、ユーザーが入力した値に対して予測のみを行う。

```text
model.py → モデルを学習する
model.joblib → 学習済みモデルを保存する
weather_api.py → 保存済みモデルを読み込んで予測する
```

これにより、Webアプリとして軽く動作し、ユーザーが入力した気象データに対して素早く降雨確率を返すことができる。

---

## API化

`weather_api.py` では、FastAPIを用いて学習済みモデルをWebアプリから利用できるようにした。

### FastAPIアプリの作成

```python
app = FastAPI()
```

FastAPIを使うことで、Pythonで作成した予測処理をWeb APIとして公開できるようにした。

---

### CORS設定

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

CORS設定を追加することで、フロントエンドのJavaScriptからAPIにリクエストを送信できるようにした。
今回は個人開発のため `allow_origins=["*"]` としているが、実運用ではアクセスを許可するURLを限定する必要がある。

---

### 静的ファイルの読み込み

```python
base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

static_path = os.path.join(project_root, "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")
```

`os.path` を用いてプロジェクトのルートディレクトリを取得し、CSSやJavaScriptなどの静的ファイルを読み込めるようにした。

`os.path` を使うことで、ローカル環境だけでなく、Render上でもファイルパスを正しく扱えるようにしている。

---

### トップページの表示

```python
@app.get("/")
def index():
    html_path = os.path.join(project_root, "templates", "weather_predict.html")
    return FileResponse(html_path)
```

トップページ `/` にアクセスしたとき、`templates/weather_predict.html` を返すように設定した。
これにより、公開URLにアクセスするとWebアプリの画面が表示される。

---

### 学習済みモデルの読み込み

```python
model_path = os.path.join(project_root, "models", "model.joblib")

try:
    result = joblib.load(model_path)
    print("\nConnected model!\n")
except FileNotFoundError:
    print(f"error: not found model.joblib: {model_path}")
    sys.exit(1)
```

API起動時に、`models/model.joblib` を読み込む。
モデルファイルが存在しない場合はエラーを表示し、処理を終了するようにした。

---

### 入力データの定義

```python
class WeatherData(BaseModel):
    AvgTemp: float
    TotalPrecip: float
    SolarHours: float
    AvgCloud: float
    vapor_pressure: float
    AvgWindSpeed: float
    MinTemp: float
```

Pydanticの `BaseModel` を用いて、APIが受け取る入力データの形式を定義した。
これにより、フロントエンドから送信されるデータの項目を明確にした。

---

### 予測API

```python
@app.post("/predict")
def predict(data: WeatherData):
```

`/predict` にPOSTリクエストが送信されると、予測処理が実行される。

受け取ったデータはPandasのDataFrameに変換する。

```python
df = pd.DataFrame([{
    "AvgTemp": data.AvgTemp,
    "TotalPrecip": data.TotalPrecip,
    "SolarHours": data.SolarHours,
    "AvgCloud": data.AvgCloud,
    "vapor_pressure": data.vapor_pressure,
    "AvgWindSpeed": data.AvgWindSpeed,
    "MinTemp": data.MinTemp
}])
```

学習時に定数項 `const` を追加していたため、API側でも同じように追加する。

```python
df.insert(0, "const", 1.0)
```

その後、保存済みモデルを用いて予測を行う。

```python
prediction = result.predict(df)
rain_probability = round(float(prediction.iloc[0] * 100), 1)
```

最後に、降雨確率をJSON形式で返す。

```python
return {
    "rain_probability": rain_probability
}
```

---

## フロントエンド

フロントエンドでは、HTML・CSS・JavaScriptを用いて、ユーザーが気象データを入力し、予測結果を確認できる画面を作成した。

---

### HTML

HTMLでは、以下の7つの入力フォームを用意した。

* 平均気温
* 降水量
* 日照時間
* 雲量
* 平均蒸気圧
* 平均風速
* 最低気温

```html
<input type="number" id="AvgTemp" placeholder="例: 23.0">
<input type="number" id="TotalPrecip" placeholder="例: 0.0">
<input type="number" id="SolarHours" placeholder="例: 6.0">
<input type="number" id="AvgCloud" placeholder="例: 8.0">
<input type="number" id="vapor_pressure" placeholder="例: 13">
<input type="number" id="AvgWindSpeed" placeholder="例: 3">
<input type="number" id="MinTemp" placeholder="例: 20.0">
```

ユーザーが「予測する」ボタンを押すと、JavaScriptの `predictWeather()` 関数が実行される。

```html
<button onclick="predictWeather()">予測する</button>
```

また、気象庁のデータ参照ページへのリンクも配置した。

---

### CSS

CSSでは、入力フォームや予測結果の表示欄を見やすく整えた。
具体的には、フォームの中央配置、背景色、余白、角丸、影などを設定している。

```css
.form {
    background: white;
    padding: 25px 30px;
    margin-top: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    width: 340px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
```

また、予測結果は初期状態では非表示にし、予測後に表示されるようにした。

```css
.hidden {
    display: none;
}
```

CSSについては、AIの支援を受けながら、ユーザーが入力しやすい画面になるように調整した。
ただし、画面構成、入力項目、予測結果の表示位置などは、アプリの目的に合わせて自分で決定した。

---

### JavaScript

JavaScriptでは、入力フォームの値を取得し、FastAPIの `/predict` にPOSTリクエストを送信している。

```javascript
const data = {
    AvgTemp: Number(document.getElementById("AvgTemp").value) || 0,
    TotalPrecip: Number(document.getElementById("TotalPrecip").value) || 0,
    SolarHours: Number(document.getElementById("SolarHours").value) || 0,
    AvgCloud: Number(document.getElementById("AvgCloud").value) || 0,
    vapor_pressure: Number(document.getElementById("vapor_pressure").value) || 0,
    AvgWindSpeed: Number(document.getElementById("AvgWindSpeed").value) || 0,
    MinTemp: Number(document.getElementById("MinTemp").value) || 0
};
```

取得したデータはJSON形式に変換し、APIへ送信する。

```javascript
const response = await fetch("/predict", {
    method: "POST",
    headers: {
        "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
});
```

APIから返ってきた `rain_probability` を受け取り、画面上に表示する。

```javascript
const result = await response.json();
const rainProb = Number(result.rain_probability);

value.textContent = rainProb.toFixed(1) + "%";
box.classList.remove("hidden");
```

また、API通信でエラーが発生した場合は、画面に「エラー」と表示するようにした。

```javascript
catch (error) {
    console.error(error);
    value.textContent = "エラー";
    box.classList.remove("hidden");
}
```

---

## Renderでのデプロイ

本アプリはRenderを用いてWeb上に公開した。

Render上でFastAPIアプリを動作させるため、必要なPythonライブラリを `requirements.txt` にまとめた。

```txt
fastapi
uvicorn
pandas
joblib
pydantic
jinja2
python-multipart
scikit-learn
statsmodels
numpy
```

Renderでは、デプロイ時に以下のコマンドで `requirements.txt` に記載されたライブラリが自動でインストールされる。

```bash
pip install -r requirements.txt
```

---

## Renderでの起動コマンド

Renderでは、Start Commandに以下を設定した。

```bash
uvicorn api.weather_api:app --host 0.0.0.0 --port $PORT
```

それぞれの意味は以下の通りである。

```text
uvicorn
→ FastAPIを動かすためのサーバー

api.weather_api:app
→ apiフォルダ内の weather_api.py に定義している app を起動する

--host 0.0.0.0
→ Render外部からアクセスできるようにする

--port $PORT
→ Renderが自動で割り当てるポート番号を使用する
```

ローカル環境では、以下のように起動できる。

```bash
uvicorn api.weather_api:app --reload
```

一方で、Renderではポート番号をRender側が指定するため、`--port $PORT` を使用している。

---

## 工夫した点

### 1. モデル構築と予測処理を分離した点

`model.py` でモデルの学習を行い、学習済みモデルを `model.joblib` として保存した。
API側では保存済みモデルを読み込んで予測のみを行う構成にしたため、Webアプリ上で毎回モデルを再学習する必要がない。

これにより、処理を軽くし、Webアプリとして利用しやすい形にした。

---

### 2. 学習時と予測時の入力形式を合わせた点

モデル学習時には、statsmodelsを使用するために `sm.add_constant()` で定数項を追加した。
そのため、API側でも以下のように `const` を追加し、学習時と同じ列構成にした。

```python
df.insert(0, "const", 1.0)
```

この処理により、保存済みモデルに対して正しい形式で入力データを渡せるようにした。

---

### 3. Webブラウザから利用できる形にした点

前回のプロジェクトでは、Pythonファイルを実行して予測を行う形だった。
今回はFastAPIとJavaScriptを連携させることで、Webブラウザから気象データを入力し、降雨確率を表示できるようにした。

---

### 4. Renderで公開した点

ローカル環境で動くだけでなく、Renderを用いてWeb上に公開した。
これにより、自分のPC以外からもアプリにアクセスできるようにした。

---

## 苦労した点

### 1. モデルファイルの読み込み

ローカル環境では動作しても、Render上ではファイルパスの違いによって `model.joblib` が読み込めないことがあった。
そのため、`os.path` を用いてプロジェクトルートからのパスを指定し、環境に依存しにくい形に修正した。

---

### 2. 学習時と予測時の列の違い

学習時には `const` を追加していたため、API側でも同じ列を追加する必要があった。
最初はこの対応が必要であることに気づきにくかったが、学習時と予測時の入力データの形をそろえることの重要性を学んだ。

---

### 3. Renderでの起動設定

Renderでは、ローカル環境と異なり、ポート番号を自分で固定するのではなく、Renderが指定する `$PORT` を使う必要があった。
そのため、以下のように起動コマンドを設定した。

```bash
uvicorn api.weather_api:app --host 0.0.0.0 --port $PORT
```

この設定により、Render上でFastAPIアプリを正しく起動できるようになった。

---

## 今後の課題

### 1. 入力値の自動取得

現在は、ユーザーが気象データを手入力する必要がある。
今後は、気象庁や外部APIから最新の気象データを自動取得し、入力の手間を減らしたい。

---

### 2. モデル精度の向上

現在はロジスティック回帰を使用しているが、今後はランダムフォレストや勾配ブースティングなど、他の機械学習モデルとの比較も行いたい。

また、前日や数日前の気象データを特徴量として追加するラグ特徴量を導入することで、より実用的な予測につなげたい。

---

### 3. 評価方法の改善

今回のモデルでは、ランダム分割を採用した。
しかし、気象データは時系列データであるため、本来は時系列を考慮した評価も重要である。

今後は、複数の `random_state` による評価や、時系列分割を改善した評価方法を検討したい。

---

### 4. UIの改善

現在の画面は、入力フォームと予測結果を表示するシンプルな構成である。
今後は、入力例の追加、スマートフォン対応、予測結果の説明文表示などを行い、より使いやすいアプリに改善したい。

---

## 学んだこと

本プロジェクトを通して、機械学習モデルを作成するだけでなく、それをWebアプリとして利用できる形にする流れを学んだ。

特に、以下の点を理解できた。

* 機械学習モデルを `joblib` で保存する理由
* 学習処理と予測処理を分ける重要性
* FastAPIを用いたAPI化の流れ
* JavaScriptからAPIへリクエストを送る方法
* RenderでFastAPIアプリを公開する方法
* ローカル環境とデプロイ環境でのパスやポート設定の違い

今回の開発により、機械学習モデルを単体で終わらせるのではなく、API化し、フロントエンドと連携させることで、実際にユーザーが利用できる形にする経験を得ることができた。

---

## まとめ

本プロジェクトでは、前回作成した翌日の降雨予測モデルをWebアプリケーションとして利用できるようにした。

具体的には、過去の気象データを用いてロジスティック回帰モデルを学習し、学習済みモデルを `model.joblib` として保存した。
その後、FastAPIを用いて予測APIを作成し、HTML・CSS・JavaScriptで作成したフロントエンドからAPIを呼び出すことで、Webブラウザ上に降雨確率を表示できるようにした。

また、Renderを用いてアプリを公開し、ローカル環境だけでなくWeb上から利用できる形にした。

今回の開発を通して、機械学習、API、フロントエンド、デプロイを一連の流れとして経験することができた。
今後は、気象データの自動取得やモデル精度の改善、UIの改善を行い、より実用的な天気予測アプリに発展させたい。
