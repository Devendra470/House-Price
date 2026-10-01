"""
train_model.py
--------------
Trains a Linear Regression model on house_price.csv and saves it as model.pkl.
Run this script once before launching the Streamlit app.
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── 1. Load data ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "house_price.csv")
df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())

# ── 2. Features & target ──────────────────────────────────────────────────────
X = df[["area", "rooms"]]
y = df["price"]

# ── 3. Train / test split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train model ────────────────────────────────────────────────────────────
model = LinearRegression()
model.fit(X_train, y_train)

# ── 5. Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print("\n-- Model Evaluation --------------------------------------------------")
print(f"  MAE  : ${mae:,.0f}")
print(f"  RMSE : ${rmse:,.0f}")
print(f"  R2   : {r2:.4f}")
print(f"  Coefficients: area={model.coef_[0]:.2f}, rooms={model.coef_[1]:.2f}")
print(f"  Intercept   : {model.intercept_:.2f}")

# -- 6. Save model -------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")
