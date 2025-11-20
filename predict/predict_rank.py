#!/usr/bin/env python3
"""
predict_weekly.py
- RMSE-only validation for weekly genre prediction
- Forecast final weeks (default next 20 weeks)
- Forecast both counts and average ranks per genre
- Per-model top-5 predicted genres
- Plots predicted vs actual for each genre and model
"""

import os
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
    grouped = df.groupby(["week","genre"]).agg(
        count=("genre","size"),
        avg_rank=("rank","mean")
    ).reset_index().sort_values(["genre","week"])
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
def series_from_weeks(weeks, values):
    ts = pd.Series(data=values, index=pd.to_datetime(weeks)).sort_index()
    full_weeks = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq='W-MON')
    ts_full = ts.reindex(full_weeks).interpolate(method='linear')
    return ts_full

# ---------------------------
# Models
# ---------------------------
def predict_exp(ts_series, steps):
    try:
        model = ExponentialSmoothing(ts_series,
                                     trend="add",
                                     seasonal="add",
                                     seasonal_periods=52,
                                     initialization_method="estimated")
        fit = model.fit()
        return np.asarray(fit.forecast(steps))
    except Exception as e:
        log(f"[ExpS] failed: {e}")
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
        log(f"[ARIMA] failed: {e}")
        return np.repeat(ts_series.iloc[-1], steps)

def predict_prophet(ts_series, steps):
    try:
        dfp = pd.DataFrame({"ds": ts_series.index, "y": ts_series.values})
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        m.fit(dfp)
        future = m.make_future_dataframe(periods=steps, freq="W-MON")
        forecast = m.predict(future)
        return np.asarray(forecast["yhat"].tail(steps).values)
    except Exception as e:
        log(f"[Prophet] failed: {e}")
        return np.repeat(ts_series.iloc[-1], steps)

# ---------------------------
# PyTorch LSTM
# ---------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size,1)
    def forward(self,x):
        out,_ = self.lstm(x)
        return self.fc(out[:, -1, :])

def lstm_forecast_pytorch(ts_series, steps, window=8, epochs=400, lr=0.0075):
    vals = np.asarray(ts_series.values, dtype=float)
    if len(vals) <= window:
        return np.repeat(vals[-1], steps)
    meanv, stdv = vals.mean(), vals.std()
    scaled = (vals - meanv)/(stdv+1e-8)
    X,Y=[],[]
    for i in range(len(scaled)-window):
        X.append(scaled[i:i+window])
        Y.append(scaled[i+window])
    X = np.array(X).reshape(-1,window,1).astype(np.float32)
    Y = np.array(Y).reshape(-1,1).astype(np.float32)

    device = torch.device("cpu")
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X_t = torch.from_numpy(X).to(device)
    Y_t = torch.from_numpy(Y).to(device)

    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        out = model(X_t)
        loss = loss_fn(out,Y_t)
        loss.backward()
        optimizer.step()

    model.eval()
    recent = scaled[-window:].tolist()
    preds_scaled=[]
    for _ in range(steps):
        x_in = np.array(recent[-window:]).reshape(1,window,1).astype(np.float32)
        x_t = torch.from_numpy(x_in).to(device)
        with torch.no_grad():
            p = model(x_t).cpu().numpy().ravel()[0]
        preds_scaled.append(p)
        recent.append(p)
    preds = np.array(preds_scaled)
    return preds*stdv + meanv

# ---------------------------
# Plots: predicted vs actual (all models)
# ---------------------------
def plot_pred_vs_actual_all_models(ts_actual, ts_train, genre, preds_dict, target="Count", steps=None, outpath="plots"):
    """
    ts_actual: pd.Series of actual values
    ts_train: pd.Series of training series (for x-axis alignment)
    preds_dict: dict of {model_name: predictions array}, already computed
    """
    os.makedirs(outpath, exist_ok=True)
    if steps is None:
        steps = len(ts_actual)

    plt.figure(figsize=(10,5))
    plt.plot(ts_actual.index, ts_actual.values, marker='o', color='black', label="Actual")
    colors = ["red", "blue", "green", "orange"]
    for i, (model, preds) in enumerate(preds_dict.items()):
        plt.plot(ts_actual.index, preds, marker='x', color=colors[i], label=f"Predicted {model}")

    plt.title(f"{genre} - {target} | Actual vs Predicted (All Models)")
    plt.xlabel("Week")
    plt.ylabel(target)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    file_name = f"{outpath}/{genre}_{target}_pred_vs_all_models.png".replace(" ","_")
    plt.savefig(file_name)
    plt.close()
    log(f"[DEBUG] Saved combined plot: {file_name}")

# ---------------------------
# Evaluate validation with plots
# ---------------------------
def evaluate_models(train_df, valid_df, plot_dir="plots"):
    genres = sorted(train_df["genre"].unique())
    log(f"Evaluating {len(genres)} genres using RMSE")
    records=[]
    for genre in genres:
        tr = train_df[train_df["genre"]==genre]
        va = valid_df[valid_df["genre"]==genre]
        if len(va)==0:
            log(f"  [SKIP] no validation data for {genre}")
            continue

        ts_train_count = series_from_weeks(tr["week"].values, tr["count"].values)
        ts_valid_count = series_from_weeks(va["week"].values, va["count"].values)
        ts_train_rank  = series_from_weeks(tr["week"].values, tr["avg_rank"].values)
        ts_valid_rank  = series_from_weeks(va["week"].values, va["avg_rank"].values)
        steps = len(ts_valid_count)

        # Get predictions for all models
        preds_count_dict = {
            "ExpS": predict_exp(ts_train_count, steps),
            "ARIMA": predict_arima(ts_train_count, steps),
            "Prophet": predict_prophet(ts_train_count, steps),
            "LSTM": lstm_forecast_pytorch(ts_train_count, steps)
        }
        log(f"Genre: {genre}: Done Count Predictions")
        preds_rank_dict = {
            "ExpS": predict_exp(ts_train_rank, steps),
            "ARIMA": predict_arima(ts_train_rank, steps),
            "Prophet": predict_prophet(ts_train_rank, steps),
            "LSTM": lstm_forecast_pytorch(ts_train_rank, steps)
        }
        log(f"Genre: {genre}: Done Rank Predictions")

        # Compute RMSE
        rmse_exp_c = compute_rmse(ts_valid_count.values, preds_count_dict["ExpS"])
        rmse_arima_c = compute_rmse(ts_valid_count.values, preds_count_dict["ARIMA"])
        rmse_prop_c = compute_rmse(ts_valid_count.values, preds_count_dict["Prophet"])
        rmse_lstm_c = compute_rmse(ts_valid_count.values, preds_count_dict["LSTM"])

        rmse_exp_r = compute_rmse(ts_valid_rank.values, preds_rank_dict["ExpS"])
        rmse_arima_r = compute_rmse(ts_valid_rank.values, preds_rank_dict["ARIMA"])
        rmse_prop_r = compute_rmse(ts_valid_rank.values, preds_rank_dict["Prophet"])
        rmse_lstm_r = compute_rmse(ts_valid_rank.values, preds_rank_dict["LSTM"])

        # --- Combined predicted vs actual plots ---
        plot_pred_vs_actual_all_models(ts_valid_count, ts_train_count, genre, preds_count_dict, target="Count", steps=steps, outpath=plot_dir)
        plot_pred_vs_actual_all_models(ts_valid_rank, ts_train_rank, genre, preds_rank_dict, target="Rank", steps=steps, outpath=plot_dir)

        log(f"Genre: {genre}: Count RMSE [ExpS:{rmse_exp_c:.2f}, ARIMA:{rmse_arima_c:.2f}, Prophet:{rmse_prop_c:.2f}, LSTM:{rmse_lstm_c:.2f}], "
            f"Rank RMSE [ExpS:{rmse_exp_r:.2f}, ARIMA:{rmse_arima_r:.2f}, Prophet:{rmse_prop_r:.2f}, LSTM:{rmse_lstm_r:.2f}]")

        records.append({
            "genre": genre,
            "Exp_RMSE_Count": rmse_exp_c,
            "ARIMA_RMSE_Count": rmse_arima_c,
            "Prophet_RMSE_Count": rmse_prop_c,
            "LSTM_RMSE_Count": rmse_lstm_c,
            "Exp_RMSE_Rank": rmse_exp_r,
            "ARIMA_RMSE_Rank": rmse_arima_r,
            "Prophet_RMSE_Rank": rmse_prop_r,
            "LSTM_RMSE_Rank": rmse_lstm_r
        })
    return pd.DataFrame.from_records(records)

# ---------------------------
# Forecast final weeks
# ---------------------------
def forecast_final_weeks(grouped_df, forecast_weeks=20):
    genres = sorted(grouped_df["genre"].unique())
    rows=[]
    for genre in genres:
        log(f"Forecasting for genre: {genre}")
        gg = grouped_df[grouped_df["genre"]==genre].copy()
        if len(gg)==0:
            continue
        ts_count = series_from_weeks(gg["week"].values, gg["count"].values)
        ts_rank  = series_from_weeks(gg["week"].values, gg["avg_rank"].values)
        steps = forecast_weeks

        exp_c = predict_exp(ts_count, steps)[-1]
        arima_c = predict_arima(ts_count, steps)[-1]
        prop_c = predict_prophet(ts_count, steps)[-1]
        lstm_c = lstm_forecast_pytorch(ts_count, steps)[-1]
        avg_c = np.mean([exp_c, arima_c, prop_c, lstm_c])

        exp_r = predict_exp(ts_rank, steps)[-1]
        arima_r = predict_arima(ts_rank, steps)[-1]
        prop_r = predict_prophet(ts_rank, steps)[-1]
        lstm_r = lstm_forecast_pytorch(ts_rank, steps)[-1]
        avg_r = np.mean([exp_r, arima_r, prop_r, lstm_r])

        rows.append({
            "Genre": genre,
            "ExponentialSmoothing_Count": float(exp_c),
            "ARIMA_Count": float(arima_c),
            "Prophet_Count": float(prop_c),
            "LSTM_Count": float(lstm_c),
            "Average_Count": float(avg_c),
            "ExponentialSmoothing_Rank": float(exp_r),
            "ARIMA_Rank": float(arima_r),
            "Prophet_Rank": float(prop_r),
            "LSTM_Rank": float(lstm_r),
            "Average_Rank": float(avg_r)
        })
    return pd.DataFrame(rows).sort_values("Average_Count", ascending=False).reset_index(drop=True)

# ---------------------------
# Top 5 per model
# ---------------------------
def top5_per_model(pred_df, target="Count"):
    models = [f"ExponentialSmoothing_{target}", f"ARIMA_{target}", f"Prophet_{target}", f"LSTM_{target}"]
    top5_dict = {}
    for model in models:
        top5_dict[model] = pred_df.sort_values(model, ascending=False).head(5)[["Genre", model]].reset_index(drop=True)
    return top5_dict

# ---------------------------
# Trend plots
# ---------------------------
def plot_trend_per_model(pred_df, outpath="trend_per_model.png"):
    # Count plot
    models_count = ["ExponentialSmoothing_Count","ARIMA_Count","Prophet_Count","LSTM_Count"]
    plt.figure(figsize=(12,6))
    for model in models_count:
        plt.plot(pred_df["Genre"], pred_df[model], marker='o', label=model)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Predicted Count")
    plt.title("Predicted Genre Counts per Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath.replace(".png","_count.png"))
    plt.close()

    # Rank plot
    models_rank = ["ExponentialSmoothing_Rank","ARIMA_Rank","Prophet_Rank","LSTM_Rank"]
    plt.figure(figsize=(12,6))
    for model in models_rank:
        plt.plot(pred_df["Genre"], pred_df[model], marker='o', label=model)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Predicted Average Rank")
    plt.title("Predicted Genre Average Rank per Model")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath.replace(".png","_rank.png"))
    plt.close()
    log(f"[DEBUG] Saved trend plots (count & rank) to {outpath.replace('.png','_count.png')} and {outpath.replace('.png','_rank.png')}")

# ---------------------------
# MAIN
# ---------------------------
def main(forecast_weeks=20, validation_weeks=52):
    df = load_dataset("dummy.csv")
    grouped = build_genre_week_counts(df)
    train_df, valid_df = train_valid_split(grouped, valid_weeks=validation_weeks)

    log("\n=== Validation using RMSE and Predicted vs Actual Plots ===")
    metrics_df = evaluate_models(train_df, valid_df)
    if not metrics_df.empty:
        print(metrics_df.to_string(index=False))

    log(f"\n=== Forecasting next {forecast_weeks} weeks ===")
    pred_final = forecast_final_weeks(grouped, forecast_weeks=forecast_weeks)
    print(pred_final.to_string(index=False))

    log("\n--- Top 5 Genres by Count ---")
    top5_count = top5_per_model(pred_final, target="Count")
    for model, df_top5 in top5_count.items():
        print(f"\nTop 5 by {model}")
        print(df_top5.to_string(index=False))

    log("\n--- Top 5 Genres by Average Rank ---")
    top5_rank = top5_per_model(pred_final, target="Rank")
    for model, df_top5 in top5_rank.items():
        print(f"\nTop 5 by {model}")
        print(df_top5.to_string(index=False))

    plot_trend_per_model(pred_final, outpath=f"trend_per_model_weekly.png")

if __name__=="__main__":
    main(forecast_weeks=20, validation_weeks=52)
