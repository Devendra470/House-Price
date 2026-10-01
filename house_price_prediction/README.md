# House Price Prediction

A machine learning project that predicts house prices based on **area (sq ft)** and **number of rooms**, using a Linear Regression model. The project includes a Streamlit frontend, a Flask REST API backend, and a standalone model training script.

---

## Project Structure

```
house_price_prediction/
├── app.py            # Streamlit frontend (interactive UI + EDA charts)
├── api.py            # Flask REST API backend (/predict endpoint)
├── train_model.py    # Trains the model and saves model.pkl
├── model.pkl         # Saved trained model (auto-generated)
├── requirements.txt  # Python dependencies
└── README.md         # This file

house_price.csv       # Dataset (in parent directory)
```

---

## Tech Stack

| Layer       | Library             |
|-------------|---------------------|
| ML Model    | scikit-learn        |
| Data        | pandas, numpy       |
| Frontend    | Streamlit           |
| Backend API | Flask               |
| Charts      | matplotlib, seaborn |
| Model Save  | joblib              |

---

## Setup

### 1. Install dependencies

```bash
pip install -r house_price_prediction/requirements.txt
```

### 2. Train the model

Run this once to generate `model.pkl`:

```bash
python house_price_prediction/train_model.py
```

Expected output:
```
Dataset loaded: 40 rows, 3 columns
-- Model Evaluation --
  MAE  : $6,067
  RMSE : $6,860
  R2   : 0.9980
Model saved to: .../model.pkl
```

---

## Running the App

### Option A — Streamlit Frontend (recommended)

```bash
streamlit run house_price_prediction/app.py
```

Opens at **http://localhost:8501**

Features:
- Sidebar: enter area & rooms, click **Predict Price**
- Tab 1: raw dataset + statistics
- Tab 2: EDA charts (scatter, boxplot, histogram, heatmap, regression line)
- Tab 3: model performance metrics (MAE, RMSE, R², residuals)

### Option B — Flask REST API

```bash
python house_price_prediction/api.py
```

Runs at **http://127.0.0.1:5000**

**Predict price (POST /predict):**

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"area": 1500, "rooms": 4}'
```

Response:
```json
{
  "area": 1500,
  "rooms": 4,
  "predicted_price": 258747.40
}
```

**Health check (GET /health):**

```bash
curl http://127.0.0.1:5000/health
```

---

## Model Details

- **Algorithm:** Linear Regression
- **Features:** `area` (sq ft), `rooms` (count)
- **Target:** `price` (USD)
- **Train/Test split:** 80% / 20%
- **R² Score:** 0.9980

---

## Dataset

`house_price.csv` contains 40 samples with columns:

| Column | Description            | Range       |
|--------|------------------------|-------------|
| area   | Property size (sq ft)  | 700 – 4200  |
| rooms  | Number of rooms        | 2 – 11      |
| price  | Sale price (USD)       | 110K – 810K |
