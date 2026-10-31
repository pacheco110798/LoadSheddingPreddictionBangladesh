import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "load_shedding"

st.set_page_config(page_title="Load Shedding — SVR", page_icon="⚡", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(os.path.join(HERE, "model", "svr_rbf.joblib"))


@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(HERE, "data", "study_period.csv"), parse_dates=["datetime"])
    df["prediction"] = load_model()["model"].predict(df.drop(columns=[TARGET, "split"]))
    return df


artifact = load_model()
model, numeric_features, params = artifact["model"], artifact["numeric_features"], artifact["params"]
df = load_data()
train, test = df[df["split"] == "train"], df[df["split"] == "test"]

st.title("⚡ Load Shedding en la red eléctrica de Bangladesh")
st.caption(
    "Support Vector Regression (kernel RBF, C = 100, γ = 0.01) · PGCB Hourly Generation Dataset + clima Open-Meteo · "
    "Universidad Autónoma de Chihuahua — Machine Learning"
)

y_true, y_pred = test[TARGET], test["prediction"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE Test", f"{np.sqrt(mean_squared_error(y_true, y_pred)):.2f} MW")
c2.metric("MAE Test", f"{mean_absolute_error(y_true, y_pred):.2f} MW")
c3.metric("R² Test", f"{r2_score(y_true, y_pred):.3f}")
c4.metric("RMSE CV (Train)", "113.42 MW")

tab_series, tab_day, tab_sim, tab_model = st.tabs(
    ["📈 Serie temporal", "🕐 Detalle por día", "🎛️ Simulador", "ℹ️ Modelo"]
)

with tab_series:
    split_choice = st.radio("Periodo", ["Test", "Train", "Todo"], horizontal=True)
    subset = {"Test": test, "Train": train, "Todo": df}[split_choice]
    daily = (
        subset.set_index("datetime")[[TARGET, "prediction"]]
        .resample("D").mean()
        .rename(columns={TARGET: "Real", "prediction": "Predicción SVR"})
    )
    st.line_chart(daily, y_label="MW (promedio diario)", color=["#d62728", "#1f1f1f"])
    st.caption(
        "Train: 2022-06-17 → 2024-11-10 · Test: 2024-11-10 → 2025-06-17. "
        "El Test muestra un cambio de régimen: el load shedding de 2025 es mucho menor que en años anteriores."
    )

with tab_day:
    dates = sorted(df["datetime"].dt.date.unique())
    day = st.date_input(
        "Día", value=test["datetime"].dt.date.iloc[0], min_value=dates[0], max_value=dates[-1]
    )
    hourly = df[df["datetime"].dt.date == day]
    if hourly.empty:
        st.info("No hay registros para ese día.")
    else:
        chart = hourly.set_index("datetime")[[TARGET, "prediction"]].rename(
            columns={TARGET: "Real", "prediction": "Predicción SVR"}
        )
        st.line_chart(chart, y_label="MW", color=["#d62728", "#1f1f1f"])
        st.dataframe(
            hourly[["datetime", "demand_mw", "generation_mw", "temperature_2m", TARGET, "prediction"]]
            .rename(columns={"prediction": "predicción"})
            .round({"demand_mw": 1, "generation_mw": 1, "temperature_2m": 1, TARGET: 1, "predicción": 1}),
            hide_index=True,
            width="stretch",
        )

with tab_sim:
    st.write("Ajusta las condiciones de una hora y obtén la predicción de load shedding del modelo.")
    defaults = train.median(numeric_only=True)
    col_a, col_b = st.columns(2)
    with col_a:
        demand = st.slider("Demanda (MW)", 5000, 18000, int(defaults["demand_mw"]), step=50)
        generation = st.slider("Generación (MW)", 5000, 18000, int(defaults["generation_mw"]), step=50)
        liquid_fuel = st.slider("Combustible líquido (MW)", 0, 6000, int(defaults["liquid_fuel"]), step=50)
        india_tripura = st.slider("Importación India–Tripura (MW)", 0, 200, int(defaults["india_tripura"]))
    with col_b:
        temperature = st.slider("Temperatura (°C)", 10.0, 40.0, float(round(defaults["temperature_2m"], 1)), step=0.5)
        apparent = st.slider("Sensación térmica (°C)", 10.0, 50.0, float(round(defaults["apparent_temperature"], 1)), step=0.5)
        hour = st.slider("Hora del día", 0, 23, 21)
        month = st.slider("Mes", 1, 12, 6)
        remarks = st.selectbox("Periodo pico", ["None", "Day_Peak", "Evening_Peak"])

    row = train.drop(columns=[TARGET, "split", "prediction"]).iloc[[-1]].copy()
    row["datetime"] = pd.Timestamp(2024, month, 15, hour)
    row["demand_mw"], row["generation_mw"] = demand, generation
    row["liquid_fuel"], row["india_tripura"] = liquid_fuel, india_tripura
    row["temperature_2m"], row["apparent_temperature"] = temperature, apparent
    row["Hour"], row["Month"] = hour, month
    row["remarks"] = np.nan if remarks == "None" else remarks
    prediction = float(model.predict(row)[0])

    st.metric("Load shedding predicho", f"{max(prediction, 0):,.0f} MW", help=f"Salida cruda del SVR: {prediction:.1f} MW")
    st.caption(f"Déficit demanda − generación: {demand - generation:,} MW")

with tab_model:
    st.subheader("Pipeline")
    st.code(
        "TransformedTargetRegressor(\n"
        "    Pipeline([\n"
        "        ColumnTransformer(\n"
        "            num:    SimpleImputer(median) + StandardScaler   -> " + ", ".join(numeric_features) + "\n"
        "            cat:    SimpleImputer('None') + OneHotEncoder     -> remarks\n"
        "            onehot: OneHotEncoder                             -> Hour, Month\n"
        "        ),\n"
        f"        SVR(kernel='{params['kernel']}', C={params['C']}, gamma={params['gamma']}),\n"
        "    ]),\n"
        "    transformer=StandardScaler(),\n"
        ")",
        language="text",
    )
    st.subheader("Comparación de modelos")
    st.dataframe(
        pd.DataFrame(
            [
                ["SVR RBF optimizado (C=100, γ=0.01)", 113.42, 0.83, 50.76, 33.22, 0.68],
                ["SVR RBF (por defecto)", 146.73, 0.68, 82.67, 59.00, 0.15],
                ["Ridge (α=100)", 327.50, 0.16, 115.59, 92.81, -0.66],
            ],
            columns=["Modelo", "RMSE CV", "R² CV", "RMSE Test", "MAE Test", "R² Test"],
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption("Validación cruzada temporal: TimeSeriesSplit(n_splits=5, gap=24) sobre Train. Partición cronológica 80/20.")
