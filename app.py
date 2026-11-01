import os

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "load_shedding"
SERIES = ["Real", "Predicción SVR"]

st.set_page_config(page_title="Load Shedding — SVR", page_icon="⚡", layout="centered")

# Tokens de notebook-theme-preview.html; los colores base vienen de .streamlit/config.toml
LIGHT = {
    "accent": "#5B3DC8", "accent-soft": "#ECE6FC", "mark": "#E3D9FF", "surface": "#F5F2FA",
    "border": "#E4DFEE", "fg": "#1F1B2D", "muted": "#686179", "tip": "#127F6E", "warn": "#A86A00",
}
DARK = {
    "accent": "#B39DFF", "accent-soft": "#2A2245", "mark": "#43357A", "surface": "#1E1A29",
    "border": "#2F2940", "fg": "#ECE8F5", "muted": "#A39BB7", "tip": "#4FCBB5", "warn": "#E3A94A",
}
tokens = DARK if st.context.theme.type == "dark" else LIGHT
css_vars = "".join(f"--nb-{k}: {v};" for k, v in tokens.items())

st.html(f"""
<style>
:root {{ {css_vars}
  --nb-font-display: "IBM Plex Mono", "Courier New", monospace; }}
.block-container {{ padding-top: 3.5rem; }}
.nb-title {{ font-family: var(--nb-font-display); font-size: clamp(2rem, 5vw, 2.9rem); font-weight: 600;
  line-height: 1.12; letter-spacing: -0.045em; word-spacing: -0.2em; margin: 0 0 1.1rem; text-wrap: balance; }}
.nb-description {{ font-size: 1.2rem; line-height: 1.5; color: var(--nb-muted); max-width: 52ch; margin: 0 0 1.5rem; }}
.nb-meta {{ display: flex; flex-wrap: wrap; gap: 0.5rem 1.5rem; padding-top: 1rem; margin: 0;
  border-top: 1px solid var(--nb-border); color: var(--nb-muted); font-size: 0.8rem; font-family: var(--nb-font-display); }}
.nb-tags {{ display: flex; flex-wrap: wrap; gap: 0.4rem; list-style: none; padding: 0 !important; margin: 1rem 0 0 !important; }}
.nb-tags li {{ font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 500; padding: 0.15rem 0.6rem;
  border-radius: 999px; background: var(--nb-accent-soft); color: var(--nb-accent); margin: 0; }}
.nb-h2 {{ font-family: var(--nb-font-display); font-size: 1.45rem; font-weight: 600; letter-spacing: -0.035em;
  word-spacing: -0.15em; margin: 2.75rem 0 0.25rem; scroll-margin-top: 4rem; color: var(--nb-fg); }}
.nb-h2 span {{ color: var(--nb-accent); margin-right: 0.6em; }}
.nb-callout {{ --kind: var(--nb-accent); margin: 1rem 0 1.5rem; padding: 0.9rem 1.15rem 0.9rem 1.1rem;
  border-left: 3px solid var(--kind); border-radius: 0 4px 4px 0; background: var(--nb-surface); }}
.nb-callout[data-kind="result"] {{ background: var(--nb-accent-soft); }}
.nb-callout[data-kind="tip"] {{ --kind: var(--nb-tip); }}
.nb-callout[data-kind="warning"] {{ --kind: var(--nb-warn); }}
.nb-callout-label {{ display: block; font-family: var(--nb-font-display); font-size: 0.75rem; font-weight: 600;
  color: var(--kind); margin-bottom: 0.25rem; }}
.nb-callout p {{ margin: 0; }}
mark {{ background: var(--nb-mark); color: inherit; padding: 0.05em 0.25em; border-radius: 2px; }}
.nb-caption {{ font-size: 0.875rem; line-height: 1.5; color: var(--nb-muted); margin: 0.25rem 0 1.5rem; max-width: 60ch; }}
.nb-caption b {{ font-family: var(--nb-font-display); font-size: 0.92em; color: var(--nb-fg); }}
.nb-table {{ border-collapse: collapse; font-size: 0.9rem; font-variant-numeric: tabular-nums; line-height: 1.45;
  border-top: 1.5px solid var(--nb-fg); border-bottom: 1.5px solid var(--nb-fg); width: 100%; margin: 0.5rem 0; }}
.nb-table th {{ text-align: right; font-weight: 600; font-family: var(--nb-font-display); font-size: 0.8rem;
  padding: 0.55rem 0.8rem 0.45rem; border-bottom: 1px solid var(--nb-fg); }}
.nb-table td {{ text-align: right; padding: 0.4rem 0.8rem; border-bottom: 1px solid var(--nb-border); }}
.nb-table th:first-child, .nb-table td:first-child {{ text-align: left; }}
.nb-table tr:last-child td {{ border-bottom: 0; }}
.nb-table tr.best td {{ font-weight: 600; }}
.nb-toc-title {{ font-family: var(--nb-font-display); font-weight: 600; font-size: 0.85rem; margin-bottom: 0.75rem; }}
.nb-toc ol {{ list-style: none; padding: 0; margin: 0; counter-reset: toc; font-family: var(--nb-font-display); font-size: 0.8rem; }}
.nb-toc li {{ counter-increment: toc; padding: 0.3rem 0 0.3rem 0.85rem; border-left: 2px solid var(--nb-border); margin: 0; }}
.nb-toc li:hover {{ border-left-color: var(--nb-accent); }}
.nb-toc a {{ color: var(--nb-muted) !important; text-decoration: none !important; }}
.nb-toc a::before {{ content: counter(toc) ". "; }}
.nb-toc a:hover {{ color: var(--nb-accent) !important; }}
.nb-refs {{ padding-left: 1.2rem !important; margin: 0.5rem 0 0 !important; line-height: 1.6; }}
.nb-refs li {{ margin-bottom: 0.6rem; }}
.nb-refs a {{ color: var(--nb-accent) !important; text-decoration: underline; text-underline-offset: 3px;
  text-decoration-color: color-mix(in srgb, var(--nb-accent) 40%, transparent); }}
[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-variant-numeric: tabular-nums; }}
[data-testid="stMetricLabel"] p {{ font-family: var(--nb-font-display); font-size: 0.78rem; color: var(--nb-muted); }}
</style>
""")

SECTIONS = [
    ("problema", "El problema"),
    ("datos", "Datos y modelo"),
    ("resultados", "Resultados"),
    ("detalle", "Detalle por día"),
    ("simulador", "Simulador"),
    ("comparacion", "Comparación de modelos"),
    ("limitaciones", "Limitaciones"),
    ("referencias", "Referencias"),
]
figure_count = 0


def heading(anchor):
    number = [a for a, _ in SECTIONS].index(anchor) + 1
    st.html(f'<h2 class="nb-h2" id="{anchor}"><span>{number}</span>{dict(SECTIONS)[anchor]}</h2>')


def callout(kind, label, text):
    st.html(f'<div class="nb-callout" data-kind="{kind}"><span class="nb-callout-label">{label}</span><p>{text}</p></div>')


def caption(text):
    global figure_count
    figure_count += 1
    st.html(f'<p class="nb-caption"><b>Figura {figure_count}.</b> {text}</p>')


def series_chart(frame, y_title):
    long = frame.reset_index().melt(id_vars="datetime", var_name="Serie", value_name="MW")
    color = alt.Color("Serie:N", sort=SERIES, scale=alt.Scale(domain=SERIES), legend=alt.Legend(orient="top", title=None))
    base = alt.Chart(long).encode(
        x=alt.X("datetime:T", title=None),
        y=alt.Y("MW:Q", title=y_title),
        color=color,
        tooltip=[alt.Tooltip("datetime:T", title="Fecha"), "Serie:N", alt.Tooltip("MW:Q", format=",.1f")],
    )
    real = base.transform_filter(alt.datum.Serie == "Real").mark_area(opacity=0.35, line=True)
    pred = base.transform_filter(alt.datum.Serie == "Predicción SVR").mark_line(strokeWidth=1.4)
    st.altair_chart(alt.layer(real, pred).properties(height=320).interactive(bind_y=False), width="stretch")


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
y_true, y_pred = test[TARGET], test["prediction"]
test_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

with st.sidebar:
    items = "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t in SECTIONS)
    st.html(f'<div class="nb-toc-title">Contenido</div><nav class="nb-toc"><ol>{items}</ol></nav>')

st.html("""
<header>
  <h1 class="nb-title">Prediciendo load shedding en la red eléctrica de Bangladesh</h1>
  <p class="nb-description">Support Vector Regression con kernel RBF sobre generación, demanda y clima horario para estimar
  los MW de carga que la red deja sin servicio.</p>
  <p class="nb-meta"><span>Andrea Rodríguez · 395699</span><span>Universidad Autónoma de Chihuahua</span><span>Machine Learning</span></p>
  <ul class="nb-tags"><li>machine learning</li><li>series de tiempo</li><li>SVR</li><li>scikit-learn</li></ul>
</header>
""")

heading("problema")
st.markdown(
    "Cuando la demanda supera la generación disponible, la Power Grid Company of Bangladesh (PGCB) recorta carga de forma "
    "programada: el **load shedding**. El objetivo es estimar cuántos MW se recortan en cada hora a partir de la operación "
    "de la red y del clima, minimizando el **RMSE**."
)
callout(
    "result", "Resultado",
    f"El SVR RBF optimizado obtiene un RMSE en Test de <mark>{test_rmse:.2f} MW</mark> y un RMSE de validación cruzada de "
    "113.42 MW, 65% menor que Ridge (327.50 MW) y 33% menor que XGBoost (169 MW).",
)

heading("datos")
st.markdown(
    "- **Dataset:** PGCB Hourly Generation Dataset (UCI) más clima horario de Open-Meteo (promedio de 8 capitales de división).\n"
    "- **Periodo:** los últimos 3 años de registros, con partición **cronológica** 80/20: "
    "Train 2022-06-17 → 2024-11-10, Test 2024-11-10 → 2025-06-17.\n"
    "- **Validación:** `TimeSeriesSplit(n_splits=5, gap=24)` sobre Train; búsqueda en rejilla de `C` y `gamma`."
)
with st.expander("Ver pipeline"):
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
        language="python",
    )

heading("resultados")
c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE Test", f"{test_rmse:.2f} MW")
c2.metric("MAE Test", f"{mean_absolute_error(y_true, y_pred):.2f} MW")
c3.metric("R² Test", f"{r2_score(y_true, y_pred):.3f}")
c4.metric("RMSE CV", "113.42 MW")

split_choice = st.segmented_control("Periodo", ["Test", "Train", "Todo"], default="Test", label_visibility="collapsed")
subset = {"Test": test, "Train": train, "Todo": df}[split_choice or "Test"]
daily = (
    subset.set_index("datetime")[[TARGET, "prediction"]]
    .resample("D").mean()
    .rename(columns={TARGET: "Real", "prediction": "Predicción SVR"})
)
series_chart(daily, "MW (promedio diario)")
caption(f"Load shedding real frente a la predicción del SVR ({split_choice or 'Test'}, promedios diarios).")

heading("detalle")
dates = sorted(df["datetime"].dt.date.unique())
day = st.date_input("Día", value=test["datetime"].dt.date.iloc[0], min_value=dates[0], max_value=dates[-1])
hourly = df[df["datetime"].dt.date == day]
if hourly.empty:
    st.info("No hay registros para ese día.")
else:
    series_chart(
        hourly.set_index("datetime")[[TARGET, "prediction"]].rename(columns={TARGET: "Real", "prediction": "Predicción SVR"}),
        "MW",
    )
    caption(f"Load shedding hora por hora el {day:%d/%m/%Y}.")
    with st.expander("Ver datos del día"):
        st.dataframe(
            hourly[["datetime", "demand_mw", "generation_mw", "temperature_2m", TARGET, "prediction"]]
            .rename(columns={"prediction": "predicción"})
            .round({"demand_mw": 1, "generation_mw": 1, "temperature_2m": 1, TARGET: 1, "predicción": 1}),
            hide_index=True,
            width="stretch",
        )

heading("simulador")
st.markdown("Ajusta las condiciones de una hora y obtén la predicción del modelo.")
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

callout(
    "result", "Predicción",
    f"Load shedding estimado: <mark>{max(prediction, 0):,.0f} MW</mark> · déficit demanda − generación: {demand - generation:,} MW "
    f"· salida cruda del SVR: {prediction:.1f} MW",
)

heading("comparacion")
comparison = [
    ("SVR RBF optimizado (C=100, γ=0.01)", 113.42, 0.83, 50.76, 33.22, 0.68),
    ("SVR RBF (por defecto)", 146.73, 0.68, 82.67, 59.00, 0.15),
    ("Ridge (α=100)", 327.50, 0.16, 115.59, 92.81, -0.66),
]
rows = "".join(
    f'<tr class="{"best" if i == 0 else ""}"><td>{name}</td><td>{a:.2f}</td><td>{b:.2f}</td>'
    f"<td>{c:.2f}</td><td>{d:.2f}</td><td>{e:.2f}</td></tr>"
    for i, (name, a, b, c, d, e) in enumerate(comparison)
)
st.html(
    '<div style="overflow-x:auto"><table class="nb-table"><thead><tr><th>Modelo</th><th>RMSE CV</th><th>R² CV</th>'
    f"<th>RMSE Test</th><th>MAE Test</th><th>R² Test</th></tr></thead><tbody>{rows}</tbody></table></div>"
)
st.html('<p class="nb-caption">Métricas en MW. CV: promedio de los 5 folds de <code>TimeSeriesSplit</code> sobre Train.</p>')

heading("limitaciones")
callout(
    "warning", "Cambio de régimen",
    "El load shedding de 2025 es mucho menor que en 2023 y 2024 (media de 33.4 MW en Test frente a 320.6 MW en Train). "
    "Por eso el RMSE de Test es menor que el de CV, sin que el modelo sea mejor.",
)
callout(
    "tip", "Predicciones negativas",
    "El 20.8% de las predicciones en Test son negativas. El simulador las recorta a 0 MW, que es el valor físicamente posible.",
)

heading("referencias")
st.html("""
<ol class="nb-refs">
  <li>Power Grid Bangladesh PLC. <i>Generación horaria (ERP)</i>.
    <a href="https://erp.powergrid.gov.bd/w/generations/view_generations" target="_blank" rel="noreferrer">erp.powergrid.gov.bd/w/generations/view_generations</a></li>
  <li>Power Grid Company of Bangladesh. <i>PGCB Hourly Generation Dataset (Bangladesh)</i>. UCI Machine Learning Repository, 2025.
    <a href="https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh)" target="_blank" rel="noreferrer">archive.ics.uci.edu/dataset/1175</a></li>
</ol>
""")
