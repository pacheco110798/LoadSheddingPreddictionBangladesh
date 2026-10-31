"""Entrena el SVR RBF optimizado de LoadSheddingSVR copy.ipynb y guarda los artefactos para la app de Streamlit.

Uso: python train_model.py
Genera:
  - model/svr_rbf.joblib   (TransformedTargetRegressor ajustado en todo Train)
  - data/study_period.csv  (periodo de estudio con features, objetivo y columna split)
"""
import os
import zipfile
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "..", "pgcb+hourly+generation+dataset+(bangladesh).zip")
WEATHER_PATH = os.path.join(HERE, "bangladesh_weather_hourly.csv")
TARGET = "load_shedding"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "apparent_temperature", "precipitation", "cloud_cover"]
BEST_C, BEST_GAMMA = 100, 0.01


def add_time_features(df):
    df = df.copy()
    df["Hour"] = df["datetime"].dt.hour
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24.0)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24.0)
    df["DayOfWeek"] = df["datetime"].dt.dayofweek
    df["Month"] = df["datetime"].dt.month
    df["DayOfYear"] = df["datetime"].dt.dayofyear
    # En Bangladesh el fin de semana es viernes y sábado
    df["IsWeekend"] = df["DayOfWeek"].isin([4, 5]).astype(int)
    return df


def add_weather_features(df, weather):
    df = df.copy()
    hourly_weather = weather.reindex(df["datetime"].dt.floor("h"))
    for var in WEATHER_VARS:
        df[var] = hourly_weather[var].to_numpy()
    return df


def main():
    with zipfile.ZipFile(ZIP_PATH) as z:
        X = pd.read_excel(z.open("PGCB_date_power_demand.xlsx"))

    X = X.drop_duplicates().reset_index(drop=True)
    X = X.sort_values("datetime").drop_duplicates(subset="datetime", keep="last").reset_index(drop=True)

    max_mw = {
        "generation_mw": 20000, "demand_mw": 20000, "load_shedding": 5000,
        "gas": 15000, "liquid_fuel": 10000, "coal": 10000, "hydro": 1000,
        "india_bheramara_hvdc": 2000, "india_tripura": 500, "india_adani": 3000,
    }
    invalid = np.zeros(len(X), dtype=bool)
    for col, limit in max_mw.items():
        invalid |= X[col] > limit
    X = X[~invalid].reset_index(drop=True)

    end_date = X["datetime"].max()
    X = X[X["datetime"] > end_date - pd.DateOffset(years=3)].reset_index(drop=True)

    X_var = X.drop(TARGET, axis=1)
    y_var = X[TARGET].copy()
    train_x, test_x, train_y, test_y = train_test_split(X_var, y_var, test_size=0.2, shuffle=False)

    weather = pd.read_csv(WEATHER_PATH, parse_dates=["time"], index_col="time")
    train_x = add_weather_features(add_time_features(train_x), weather)
    test_x = add_weather_features(add_time_features(test_x), weather)

    train_coverage = train_x.select_dtypes(include=np.number).notna().mean()
    sparse_features = train_coverage[train_coverage < 0.5].index.tolist()

    categorical_features = ["remarks"]
    onehot_features = ["Hour", "Month"]
    numeric_features = [
        col for col in train_x.columns
        if col not in ["datetime"] + categorical_features + onehot_features + sparse_features
    ]
    train_corr = pd.concat([train_x[numeric_features], train_y], axis=1).corr()[TARGET].drop(TARGET)
    weak_features = train_corr[train_corr.abs() < 0.3].index.tolist()
    numeric_features = [col for col in numeric_features if col not in weak_features]
    print("Numéricas:", numeric_features)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_features),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), onehot_features),
    ])
    model = TransformedTargetRegressor(
        regressor=Pipeline([
            ("preprocessing", preprocessor),
            ("model", SVR(kernel="rbf", C=BEST_C, gamma=BEST_GAMMA, cache_size=1000)),
        ]),
        transformer=StandardScaler(),
    )
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    rmse = np.sqrt(np.mean((test_y - pred) ** 2))
    print(f"RMSE Test: {rmse:.2f}")

    os.makedirs(os.path.join(HERE, "model"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    joblib.dump(
        {"model": model, "numeric_features": numeric_features,
         "params": {"kernel": "rbf", "C": BEST_C, "gamma": BEST_GAMMA}},
        os.path.join(HERE, "model", "svr_rbf.joblib"),
        compress=3,
    )

    study = pd.concat([
        train_x.assign(**{TARGET: train_y, "split": "train"}),
        test_x.assign(**{TARGET: test_y, "split": "test"}),
    ]).reset_index(drop=True)
    study.to_csv(os.path.join(HERE, "data", "study_period.csv"), index=False)
    print("Artefactos guardados en model/ y data/")


if __name__ == "__main__":
    main()
