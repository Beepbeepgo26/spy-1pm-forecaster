import joblib
from datetime import datetime
import pytz
import requests
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from config import (
    ALPACA_API_KEY_ID,
    ALPACA_API_SECRET_KEY,
    ALPACA_BASE_URL,
    SYMBOL,
    MODEL_PATH,
    TARGET_TIME_PT,
)

PT = pytz.timezone("America/Los_Angeles")

app = FastAPI(title="SPY 1pm Forecaster")


class PredictionRequest(BaseModel):
    # symbol is optional; if not provided we fall back to SYMBOL from config.py
    symbol: Optional[str] = None


# Load model payload at startup
model_payload = joblib.load(MODEL_PATH)
model = model_payload["model"]
feature_columns = model_payload["feature_columns"]
residual_std = model_payload["residual_std"]


def fetch_today_bars(symbol: str) -> pd.DataFrame:
    now_pt = datetime.now(PT)
    today = now_pt.date()

    # From 6:30am PT to now
    start = PT.localize(datetime.combine(today, datetime.strptime("06:30", "%H:%M").time()))

    url = f"{ALPACA_BASE_URL}/v2/stocks/{symbol}/bars"
    params = {
        "timeframe": "1Min",
        "start": start.astimezone(pytz.utc).isoformat(),
        "end": now_pt.astimezone(pytz.utc).isoformat(),
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

    df = df.rename(
        columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
        }
    )
    return df[["open", "high", "low", "close", "volume"]]


def make_features_live(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from today's bars so far
    using the same logic as in training (but snapshot at 'now').
    """
    if df.empty:
        raise ValueError("No intraday data available yet.")

    # Clip to RTH
    df = df.between_time("06:30", TARGET_TIME_PT)
    if df.empty:
        raise ValueError("No RTH data available yet.")

    df_snap = df.copy()

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

    feats = pd.DataFrame(
        [
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
        ]
    )

    # Align columns to training columns
    for col in feature_columns:
        if col not in feats.columns:
            feats[col] = 0.0
    feats = feats[feature_columns]

    return feats


@app.post("/predict_1pm")
def predict_1pm(req: PredictionRequest):
    symbol = req.symbol or SYMBOL

    try:
        df_today = fetch_today_bars(symbol)
        if df_today.empty:
            raise HTTPException(status_code=400, detail="No intraday data yet for today.")

        feats = make_features_live(df_today)
        y_hat = float(model.predict(feats)[0])

        # simple 80% interval ~ +/- 1.28 * residual_std (rough Gaussian)
        z_80 = 1.28
        lower_80 = y_hat - z_80 * residual_std
        upper_80 = y_hat + z_80 * residual_std

        last_price = float(df_today["close"].iloc[-1])

        now_pt = datetime.now(PT)
        minutes_to_1pm = (
            datetime.combine(now_pt.date(), datetime.strptime(TARGET_TIME_PT, "%H:%M").time())
            .replace(tzinfo=PT)
            - now_pt
        ).total_seconds() / 60.0

        return {
            "symbol": symbol,
            "now_timestamp_pt": now_pt.isoformat(),
            "price_now": last_price,
            "predicted_price_1pm": y_hat,
            "lower_80": lower_80,
            "upper_80": upper_80,
            "minutes_to_1pm": minutes_to_1pm,
            "model_version": "xgb_spy_1pm_v1",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

