#!/usr/bin/env python3
"""
predict_weekly.py
- RMSE-only validation for weekly genre prediction
- Forecast final week (default last week + n weeks)
- Per-model top-5 predicted genres
"""

import warnings
import logging
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore', ConvergenceWarning)
# Disable Prophet logging
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# ---------------------------
# Logging
# ---------------------------
def log(msg: str):
    print(f"[DEBUG] {msg}")

# ---------------------------
# Load & preprocess
# ---------------------------
def load_dataset(path="dummy.csv"):
    log(f"Loading dataset from {path} ...")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["week"] = df["date"].dt.to_period('W').apply(lambda r: r.start_time)
    log(f"Loaded {len(df)} rows, weeks {df['week'].min()}-{df['week'].max()}")
    return df

def build_genre_week_counts(df):
    grouped = df.groupby(["week", "genre"]).size().reset_index(name="count").sort_values(["genre","week"])
    log(f"Aggregated to {len(grouped)} (week, genre) rows")
    return grouped

# ---------------------------
# Train/validation split
# ---------------------------
def train_valid_split(grouped_df, valid_weeks=52):
    max_week = grouped_df["week"].max()
    split_week = max_week - pd.Timedelta(weeks=valid_weeks)
    train_df = grouped_df[grouped_df["week"] <= split_week].copy()
    valid_df = grouped_df[grouped_df["week"] > split_week].copy()
    log(f"Train rows: {len(train_df)}, Valid rows: {len(valid_df)}")
    return train_df, valid_df

# ---------------------------
# RMSE metric
# ---------------------------
def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ---------------------------
# Time series helpers
# ---------------------------
def series_from_weeks(weeks, counts):
    ts = pd.Series(data=counts, index=pd.to_datetime(weeks))
    ts = ts.sort_index()
    # full weekly range
    full_weeks = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='W-MON')
    ts_full = ts.reindex(full_weeks, fill_value=0)
    missing_weeks = ts_full[ts_full==0].index.difference(ts.index)
    if len(missing_weeks) > 0:
        log(f"[DEBUG] {len(missing_weeks)} missing weeks filled with 0:\n {missing_weeks}")
    return ts_full

# ---------------------------
# Models
# ---------------------------
def predict_exp(ts_series, steps):
    try:
        ts_series.to_csv('my_data.csv', index=False)
        model = ExponentialSmoothing(ts_series, trend="add", seasonal=None, initialization_method="estimated")
        fit = model.fit()
        return np.asarray(fit.forecast(steps))
    except Exception as e:
        log(f"    [ExpS] failed: {e}")
        return np.repeat(ts_series.iloc[-1], steps)

def predict_arima(ts_series, steps):
    try:
        order = (1,1,1) if len(ts_series)<10 else (3,1,2)
        model = ARIMA(ts_series, order=order)
        fit = model.fit()
        start = len(ts_series)
        end = start + steps - 1
        return np.asarray(fit.predict(start=start, end=end))
    except Exception as e:
        log(f"    [ARIMA] failed: {e}")
        return np.repeat(ts_series.iloc[-1], steps)

def predict_prophet(ts_series, steps):
    try:
        dfp = pd.DataFrame({"ds": ts_series.index, "y": ts_series.values})
        m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=True)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=steps, freq="W-MON")
        forecast = m.predict(future)
        return np.asarray(forecast["yhat"].tail(steps).values)
    except Exception as e:
        log(f"    [Prophet] failed: {e}")
        return np.repeat(ts_series.iloc[-1], steps)

# PyTorch LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

def lstm_forecast_pytorch(ts_series, steps, window=4, epochs=200, lr=0.01):
    vals = np.asarray(ts_series.values, dtype=float)
    if len(vals) <= window:
        return np.repeat(vals[-1], steps)
    minv, maxv = vals.min(), vals.max()
    scaled = (vals-minv)/(maxv-minv) if maxv-minv>1e-8 else np.zeros_like(vals)
    X,Y = [],[]
    for i in range(len(scaled)-window):
        X.append(scaled[i:i+window])
        Y.append(scaled[i+window])
    X = np.array(X).reshape(-1,window,1).astype(np.float32)
    Y = np.array(Y).reshape(-1,1).astype(np.float32)

    device = torch.device("cpu")
    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = loss_fn(out, Y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    recent = scaled[-window:].tolist()
    preds_scaled = []
    for _ in range(steps):
        x_in = np.array(recent[-window:]).reshape(1,window,1).astype(np.float32)
        x_t = torch.from_numpy(x_in).to(device)
        with torch.no_grad():
            p = model(x_t).cpu().numpy().ravel()[0]
        preds_scaled.append(p)
        recent.append(p)
    preds = np.array(preds_scaled)
    preds_inv = preds*(maxv-minv)+minv if maxv-minv>1e-8 else np.repeat(vals[-1], steps)
    return preds_inv

# ---------------------------
# Evaluate validation
# ---------------------------
def evaluate_models(train_df, valid_df):
    genres = sorted(train_df["genre"].unique())
    log(f"Evaluating {len(genres)} genres using RMSE")
    records=[]
    for genre in genres:
        tr = train_df[train_df["genre"]==genre]
        va = valid_df[valid_df["genre"]==genre]
        if len(va)==0:
            log(f"  [SKIP] no validation data for {genre}")
            continue
        ts_train = series_from_weeks(tr["week"].values, tr["count"].values)
        ts_valid = series_from_weeks(va["week"].values, va["count"].values)
        steps = len(ts_valid)

        rmse_exp = compute_rmse(ts_valid.values, predict_exp(ts_train, steps))
        rmse_arima = compute_rmse(ts_valid.values, predict_arima(ts_train, steps))
        rmse_prop = compute_rmse(ts_valid.values, predict_prophet(ts_train, steps))
        rmse_lstm = compute_rmse(ts_valid.values, lstm_forecast_pytorch(ts_train, steps))

        log(f"  {genre}: ExpS RMSE={rmse_exp:.2f}, ARIMA RMSE={rmse_arima:.2f}, Prophet RMSE={rmse_prop:.2f}, LSTM RMSE={rmse_lstm:.2f}")

        records.append({
            "genre": genre,
            "Exp_RMSE": rmse_exp,
            "ARIMA_RMSE": rmse_arima,
            "Prophet_RMSE": rmse_prop,
            "LSTM_RMSE": rmse_lstm
        })
    return pd.DataFrame.from_records(records)

# ---------------------------
# Forecast final week
# ---------------------------
def forecast_final_weeks(grouped_df, forecast_weeks=20):
    genres = sorted(grouped_df["genre"].unique())
    rows=[]
    full_weeks = pd.date_range(start=grouped_df["week"].min(), end=grouped_df["week"].max(), freq='W-MON')
    for genre in genres:
        gg = grouped_df[grouped_df["genre"]==genre].copy()
        if len(gg)==0:
            continue
        ts_full = series_from_weeks(gg["week"].values, gg["count"].values)
        steps = forecast_weeks
        exp_v = predict_exp(ts_full, steps)[-1]
        arima_v = predict_arima(ts_full, steps)[-1]
        prop_v = predict_prophet(ts_full, steps)[-1]
        lstm_v = lstm_forecast_pytorch(ts_full, steps)[-1]
        avg = np.mean([exp_v, arima_v, prop_v, lstm_v])
        rows.append({
            "Genre": genre,
            "ExponentialSmoothing": float(exp_v),
            "ARIMA": float(arima_v),
            "Prophet": float(prop_v),
            "LSTM_PyTorch": float(lstm_v),
            "Average_Prediction": float(avg)
        })
    df_pred = pd.DataFrame(rows).sort_values("Average_Prediction", ascending=False).reset_index(drop=True)
    return df_pred

def top5_per_model(pred_df):
    models = ["ExponentialSmoothing", "ARIMA", "Prophet", "LSTM_PyTorch"]
    top5_dict = {}
    for model in models:
        top5_dict[model] = pred_df.sort_values(model, ascending=False).head(5)[["Genre", model]].reset_index(drop=True)
    return top5_dict

# ---------------------------
# Plot helper
# ---------------------------
def plot_trend_per_model(pred_df, outpath="trend_per_model.png"):
    models = ["ExponentialSmoothing", "ARIMA", "Prophet", "LSTM_PyTorch"]
    plt.figure(figsize=(12, 6))
    for model in models:
        plt.plot(pred_df["Genre"], pred_df[model], marker='o', label=model)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Genre")
    plt.ylabel("Predicted Count")
    plt.title("Predicted Genre Counts per Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()
    log(f"Saved trend plot per model to {outpath}")

# ---------------------------
# MAIN
# ---------------------------
def main(forecast_weeks=20, validation_weeks=52):
    df = load_dataset("dummy.csv")
    grouped = build_genre_week_counts(df)
    train_df, valid_df = train_valid_split(grouped, valid_weeks=validation_weeks)

    log("\n=== Validation using RMSE ===")
    metrics_df = evaluate_models(train_df, valid_df)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))

    log(f"\n=== Forecasting next {forecast_weeks} weeks ===")
    pred_final = forecast_final_weeks(grouped, forecast_weeks=forecast_weeks)
    print(pred_final.to_string(index=False))

    top5_models = top5_per_model(pred_final)
    for model, df_top5 in top5_models.items():
        print(f"\n--- Top 5 genres by {model} ---")
        print(df_top5.to_string(index=False))

    plot_trend_per_model(pred_final, outpath=f"trend_per_model_weekly.png")

if __name__=="__main__":
    main(forecast_weeks=20, validation_weeks=52)
