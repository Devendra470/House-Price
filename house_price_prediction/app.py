"""
app.py
------
Streamlit frontend for House Price Prediction.
Run with:  streamlit run app.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE_DIR, "..", "house_price.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

# ── Load / train model ────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    # Auto-train if model.pkl not found
    df = load_data()
    X  = df[["area", "rooms"]]
    y  = df["price"]
    m  = LinearRegression()
    m.fit(X, y)
    joblib.dump(m, MODEL_PATH)
    return m

df    = load_data()
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://img.icons8.com/color/96/000000/home--v1.png",
    width=80,
)
st.sidebar.title("House Price Predictor")
st.sidebar.markdown("---")
st.sidebar.markdown("**Enter property details to predict its price.**")

area  = st.sidebar.number_input(
    "🏗️ Area (sq ft)",
    min_value=int(df["area"].min()),
    max_value=10000,
    value=1500,
    step=50,
)
rooms = st.sidebar.number_input(
    "🛏️ Number of Rooms",
    min_value=int(df["rooms"].min()),
    max_value=20,
    value=4,
    step=1,
)

predict_btn = st.sidebar.button("💰 Predict Price", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small>Model: Linear Regression | Data: house_price.csv</small>",
    unsafe_allow_html=True,
)

# ── Main header ───────────────────────────────────────────────────────────────
st.title("🏠 House Price Prediction")
st.markdown(
    "Predict house prices based on **area** and **number of rooms** using a "
    "Linear Regression model trained on our dataset."
)
st.markdown("---")

# ── Prediction result ─────────────────────────────────────────────────────────
if predict_btn:
    pred = model.predict(pd.DataFrame({"area": [area], "rooms": [rooms]}))[0]
    pred = max(pred, 0)

    st.success(f"### 💵 Predicted Price: **${pred:,.0f}**")
    col1, col2, col3 = st.columns(3)
    col1.metric("Area (sq ft)", f"{area:,}")
    col2.metric("Rooms", rooms)
    col3.metric("Estimated Price", f"${pred:,.0f}")
    st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📈 EDA & Charts", "🤖 Model Performance"])

# ── Tab 1 : Dataset Overview ──────────────────────────────────────────────────
with tab1:
    st.subheader("Raw Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", len(df))
    c2.metric("Min Price", f"${df['price'].min():,}")
    c3.metric("Max Price", f"${df['price'].max():,}")

# ── Tab 2 : EDA & Charts ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Exploratory Data Analysis")

    col_a, col_b = st.columns(2)

    # Scatter: Area vs Price
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x="area", y="price", hue="rooms",
                        palette="viridis", s=80, ax=ax)
        ax.set_title("Area vs Price (coloured by Rooms)")
        ax.set_xlabel("Area (sq ft)")
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K")
        )
        st.pyplot(fig)
        plt.close(fig)

    # Scatter: Rooms vs Price
    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="rooms", y="price", palette="Set2", ax=ax)
        ax.set_title("Rooms vs Price Distribution")
        ax.set_xlabel("Number of Rooms")
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K")
        )
        st.pyplot(fig)
        plt.close(fig)

    col_c, col_d = st.columns(2)

    # Histogram: Price distribution
    with col_c:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df["price"], bins=12, kde=True, color="#3b82d4", ax=ax)
        ax.set_title("Price Distribution")
        ax.set_xlabel("Price ($)")
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K")
        )
        st.pyplot(fig)
        plt.close(fig)

    # Correlation heatmap
    with col_d:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax,
        )
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)

    # Regression line: Area vs Price
    st.subheader("Regression Fit — Area vs Price")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.regplot(data=df, x="area", y="price", scatter_kws={"s": 60},
                line_kws={"color": "red"}, ax=ax)
    ax.set_title("Linear Regression Fit: Area vs Price")
    ax.set_xlabel("Area (sq ft)")
    ax.set_ylabel("Price ($)")
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K")
    )
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 3 : Model Performance ─────────────────────────────────────────────────
with tab3:
    st.subheader("Model Evaluation Metrics")

    X     = df[["area", "rooms"]]
    y_col = df["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_col, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    m1, m2, m3 = st.columns(3)
    m1.metric("MAE",  f"${mae:,.0f}")
    m2.metric("RMSE", f"${rmse:,.0f}")
    m3.metric("R² Score", f"{r2:.4f}")

    st.markdown(
        f"""
        | Metric | Value | Meaning |
        |--------|-------|---------|
        | **MAE** | ${mae:,.0f} | Average absolute error per prediction |
        | **RMSE** | ${rmse:,.0f} | Root mean squared error (penalises large errors) |
        | **R²** | {r2:.4f} | {r2*100:.1f}% variance explained by the model |
        """
    )

    # Actual vs Predicted
    col_e, col_f = st.columns(2)

    with col_e:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(y_test, y_pred, color="#3b82d4", edgecolors="white",
                   s=70, alpha=0.85)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect fit")
        ax.set_xlabel("Actual Price ($)")
        ax.set_ylabel("Predicted Price ($)")
        ax.set_title("Actual vs Predicted Price")
        ax.legend()
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:.0f}K"))
        st.pyplot(fig)
        plt.close(fig)

    with col_f:
        residuals = y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(residuals, bins=10, kde=True, color="#7c5cd8", ax=ax)
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Residuals Distribution")
        ax.set_xlabel("Residual ($)")
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Model Coefficients")
    coef_df = pd.DataFrame({
        "Feature":     ["area", "rooms"],
        "Coefficient": model.coef_,
    })
    st.dataframe(coef_df, use_container_width=True)
    st.info(f"**Intercept:** {model.intercept_:,.2f}")
