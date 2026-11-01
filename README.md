# Load Shedding Prediction — Bangladesh

Predict hourly **load shedding** (MW) on Bangladesh's national power grid with Support Vector Regression.

## Data

- **[PGCB Hourly Generation Dataset (Bangladesh)](https://archive.ics.uci.edu/dataset/1175/pgcb+hourly+generation+dataset+(bangladesh))**, UCI Machine Learning Repository. Hourly records from the Power Grid Company of Bangladesh, April 2015 to June 2025: generation, demand, load shedding, generation by fuel source and imports.
- **Hourly weather from [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api)**, averaged over Bangladesh's 8 division capitals.
- **Study period:** the last 3 years of records (June 2022 to June 2025), split 80/20 in time order.

## Task

- **Type:** regression. The target is `load_shedding`, in MW.
- **Model:** SVR with RBF kernel (C = 100, γ = 0.01). It was tuned with `GridSearchCV` and `TimeSeriesSplit` and compared against linear and polynomial SVR, Ridge and XGBoost.
- **Test results:** RMSE 50.76 MW, MAE 33.22 MW, R² 0.68.

## Run the app

```bash
pip install -r requirements.txt
python train_model.py   # optional: retrain and save the model
streamlit run app.py
```
