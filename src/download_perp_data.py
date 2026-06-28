# -*- coding: utf-8 -*-
"""
download_perp_data.py — 任務 1a:下載 BTCUSDT 永續合約的 1h K 線 + 資金費率。

注意:用的是「期貨/永續」API(fapi.binance.com),不是現貨。
- 1h OHLCV:完整歷史可抓。
- 資金費率(funding rate):每 8 小時一筆,完整歷史可抓。
- 未平倉量(OI):Binance 免費只給最近約 30 天,不夠做 2.5 年研究,先不抓
  (要長歷史得用 Coinglass 訂閱抓 1h 版)。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\download_perp_data.py
"""

import time
from datetime import datetime, timezone

import pandas as pd
import requests

import config

KLINES_LIMIT = 1000
FUNDING_LIMIT = 1000
INTERVAL_MS = {"15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def date_to_ms(date_text: str) -> int:
    return int(datetime.strptime(date_text, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_perp_klines(symbol, interval, start_ms, end_ms):
    """分批抓永續 1h K 線(邏輯同現貨:用上一批最後時間 + 一根當下批起點)。"""
    step = INTERVAL_MS[interval]
    rows, cursor = [], start_ms
    while cursor < end_ms:
        resp = requests.get(config.BINANCE_FUTURES_URL + "/fapi/v1/klines",
                            params={"symbol": symbol, "interval": interval,
                                    "startTime": cursor, "endTime": end_ms,
                                    "limit": KLINES_LIMIT}, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1][0] + step
        print(f"  K 線累計 {len(rows)} 根")
        time.sleep(0.2)

    columns = ["open_time", "open", "high", "low", "close", "volume",
               "close_time", "quote_volume", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(rows, columns=columns)[["open_time", "open", "high", "low", "close", "volume"]]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")     # 一律存 UTC 無時區,方便對齊
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)


def fetch_funding(symbol, start_ms, end_ms):
    """分批抓資金費率歷史(每 8 小時一筆)。"""
    rows, cursor = [], start_ms
    while cursor < end_ms:
        resp = requests.get(config.BINANCE_FUTURES_URL + "/fapi/v1/fundingRate",
                            params={"symbol": symbol, "startTime": cursor,
                                    "endTime": end_ms, "limit": FUNDING_LIMIT}, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        cursor = batch[-1]["fundingTime"] + 1
        time.sleep(0.2)
        if len(batch) < FUNDING_LIMIT:
            break

    df = pd.DataFrame(rows)
    df["funding_time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["funding_time", "funding_rate"]].drop_duplicates("funding_time") \
             .sort_values("funding_time").reset_index(drop=True)


def main():
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_ms = date_to_ms(config.HISTORY_START_DATE)
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"下載 {config.SYMBOL} 永續 {config.INTERVAL} K 線,自 {config.HISTORY_START_DATE}")
    klines = fetch_perp_klines(config.SYMBOL, config.INTERVAL, start_ms, end_ms)
    klines.to_csv(config.RAW_KLINES_PATH, index=False)
    print(f"K 線 {len(klines)} 根  ({klines['open_time'].iloc[0]} ~ {klines['open_time'].iloc[-1]})\n")

    print("下載資金費率歷史(每 8h 一筆)")
    funding = fetch_funding(config.SYMBOL, start_ms, end_ms)
    funding.to_csv(config.RAW_FUNDING_PATH, index=False)
    print(f"資金費率 {len(funding)} 筆  ({funding['funding_time'].iloc[0]} ~ {funding['funding_time'].iloc[-1]})")


if __name__ == "__main__":
    main()
