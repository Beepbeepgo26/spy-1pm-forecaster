import os
import requests
from datetime import datetime, timedelta
import pytz

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import joblib

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_BASE_URL,
    SYMBOL,
    MODEL_PATH,
    SNAPSHOT_TIME_PT,
    TARGET_TIME_PT,
)

PT = pytz.timezone("America/Los_Angeles")


def fetch_intraday_bars(symbol: str, start: datetime, end: datetime, timeframe: str = "1Min") -> pd.DataFrame:
    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars"

    params = {
        "timeframe": timeframe,
        "start": start.astimezone(pytz.utc).isoformat(),
        "end": end.astimezone(pytz.utc).isoformat(),
        "limit": 10000,
        "adjustment": "raw",
    }

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY_ID,
        "APCA-API-SECRET-KEY": ALPACA_API_SECRET_KEY,
    }

    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    bars = data.get("bars", [])
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df["t"] = pd.to_datetime(df["t"])
    df["t_pt"] = df["t"].dt.tz_convert(PT)
    df.set_index("t_pt", inplace=True)
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def make_features_for_day(df_day: pd.DataFrame, snapshot_time_str: str):
    if df_day.empty:
        return None, None

    df = df_day.between_time("06:30", TARGET_TIME_PT)
    if df.empty:
        return None, None

    df_snap = df.between_time("06:30", snapshot_time_str)
    if df_snap.empty:
        return None, None

    df_target = df.between_time(TARGET_TIME_PT, TARGET_TIME_PT)
    if df_target.empty:
        return None, None

    target_close = df_target["close"].iloc[-1]

    last_close = df_snap["close"].iloc[-1]
    first_open = df_snap["open"].iloc[0]

    df_snap["return_1"] = df_snap["close"].pct_change()

    return_since_open = last_close / first_open - 1.0
    high_so_far = df_snap["high"].max()
    low_so_far = df_snap["low"].min()
    day_range = max(high_so_far - low_so_far, 1e-6)
    range_location = (last_close - low_so_far) / day_range
    realized_vol = df_snap["return_1"].std()

    vwap = (df_snap["close"] * df_snap["volume"]).cumsum() / df_snap["volume"].cumsum()
    vwap_now = vwap.iloc[-1]
    above_vwap = float(last_close - vwap_now)

    ema9 = df_snap["close"].ewm(span=9, adjust=False).mean()
    ema20 = df_snap["close"].ewm(span=20, adjust=False).mean()
    ema9_now = ema9.iloc[-1]
    ema20_now = ema20.iloc[-1]

    dist_ema9 = last_close - ema9_now
    dist_ema20 = last_close - ema20_now

    minutes_since_open = (df_snap.index[-1] - df_snap.index[0]).total_seconds() / 60.0

    features = pd.Series(
        {
            "last_close": last_close,
            "return_since_open": return_since_open,
            "high_so_far": high_so_far,
            "low_so_far": low_so_far,
            "range_location": range_location,
            "day_range": day_range,
            "realized_vol": realized_vol if not np.isnan(realized_vol) else 0.0,
            "vwap_now": vwap_now,
            "above_vwap": above_vwap,
            "ema9_now": ema9_now,
            "ema20_now": ema20_now,
            "dist_ema9": dist_ema9,
            "dist_ema20": dist_ema20,
            "minutes_since_open": minutes_since_open,
        }
    )

    return features, target_close


def generate_training_data(symbol: str, num_days: int = 200):
    today = datetime.now(PT).date()
    X_rows = []
    y_vals = []

    days_checked = 0
    days_collected = 0

    while days_collected < num_days and days_checked < num_days * 3:
        day = today - timedelta(days=days_checked + 1)
        days_checked += 1

        if day.weekday() >= 5:
            continue

        start = PT.localize(datetime.combine(day, datetime.strptime("06:30", "%H:%M").time()))
        end = PT.localize(datetime.combine(day, datetime.strptime(TARGET_TIME_PT, "%H:%M").time()))

        print(f"Fetching {symbol} for {day} ...")
        df_day = fetch_intraday_bars(symbol, start, end, timeframe="1Min")
        if df_day.empty:
            print(f"  No data for {day}, skipping.")
            continue

        feats, target_close = make_features_for_day(df_day, SNAPSHOT_TIME_PT)
        if feats is None:
            print(f"  Not enough data for {day}, skipping.")
            continue

        X_rows.append(feats)
        y_vals.append(target_close)
        days_collected += 1
        print(f"  Added day {day}. Total days: {days_collected}")

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_vals, name="target_1pm")
    return X, y


def train_and_save_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    X, y = generate_training_data(SYMBOL, num_days=200)
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    rmse = mse ** 0.5
    print(f"Validation RMSE: {rmse:.2f}")

    residuals = y_valid - y_pred
    resid_std = float(residuals.std())
    print(f"Residual std (for intervals): {resid_std:.2f}")

    payload = {
        "model": model,
        "feature_columns": X.columns.tolist(),
        "residual_std": resid_std,
    }

    joblib.dump(payload, MODEL_PATH)
    print(f"Saved model payload to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()
