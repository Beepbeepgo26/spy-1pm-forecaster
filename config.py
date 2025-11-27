import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY_ID = os.getenv("ALPACA_API_KEY_ID")
ALPACA_API_SECRET_KEY = os.getenv("ALPACA_API_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")
SYMBOL = os.getenv("SYMBOL", "SPY")

# Trading hours (Pacific Time)
MARKET_OPEN_PT = "06:30"
TARGET_TIME_PT = "13:00"  # 1:00pm
SNAPSHOT_TIME_PT = "11:00"  # use 11:00 snapshot to predict 13:00

MODEL_PATH = "model/spy_1pm_model.pkl"
