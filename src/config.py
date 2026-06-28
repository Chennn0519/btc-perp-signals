# -*- coding: utf-8 -*-
"""
config.py — 全專案共用設定(BTC 永續合約訊號研究)。

這個專案的目標跟 btc-15m-lstm 不同:不是訓練模型預測價格,而是先用「統計檢定」
驗證『某些衍生品特徵,在大行情發生前是不是真的不一樣』——也就是先找「能被證明
有效的訊號」,再談模型。
"""

from pathlib import Path

# ---- 路徑 ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RAW_KLINES_PATH = DATA_RAW_DIR / "btcusdt_perp_1h.csv"
RAW_FUNDING_PATH = DATA_RAW_DIR / "btcusdt_funding.csv"
FEATURE_TABLE_PATH = DATA_PROCESSED_DIR / "feature_table.csv"
EVENT_STUDY_PATH = RESULTS_DIR / "event_study.csv"

# ---- 資料來源(注意:用的是「期貨/永續」API,不是現貨)----------------
BINANCE_FUTURES_URL = "https://fapi.binance.com"
SYMBOL = "BTCUSDT"
INTERVAL = "1h"
HISTORY_START_DATE = "2024-01-01"

# ---- 事件定義 ------------------------------------------------------------
# 「事件」= 未來 HORIZON_HOURS 小時內,價格(相對現在收盤)出現 ±MOVE_THRESHOLD 的移動。
# 這是 schumi 第一任務指定的:未來 48 小時內 ±3%。
MOVE_THRESHOLD = 0.03
HORIZON_HOURS = 48
