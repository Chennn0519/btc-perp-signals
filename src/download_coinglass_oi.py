# -*- coding: utf-8 -*-
"""
download_coinglass_oi.py — 任務 1d:用 Coinglass 抓 BTC 的未平倉量(OI)歷史。

為什麼要 Coinglass:Binance 免費只給最近約 30 天的 OI,做長期研究不夠。
間隔用 config.OI_INTERVAL(預設 4h)。注意:Coinglass HOBBYIST 方案不支援 1h,
只支援 4h/6h(1h 要升級 STANDARD);4h 對「未來 48h ±3%」這種研究已足夠。

金鑰來源(都不寫進程式、不進 git):
  1. 優先讀檔案 secrets/coinglass_api_key.txt(已被 gitignore)。
  2. 其次讀環境變數 COINGLASS_API_KEY。

實際能回溯多久取決於你的方案;程式會抓到能抓的最早,並回報真實範圍。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\download_coinglass_oi.py
"""

import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

import config

OI_HISTORY_ENDPOINT = "/api/futures/open-interest/history"
PAGE_LIMIT = 1000
SLEEP_SECONDS = 2.1        # Coinglass 限流:約 30 次/分,留安全邊際
MAX_PAGES = 60             # 上限保護,避免無限迴圈


def load_api_key() -> str:
    key_file = config.PROJECT_ROOT / "secrets" / "coinglass_api_key.txt"
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    return os.environ.get("COINGLASS_API_KEY", "").strip()


def date_to_ms(date_text: str) -> int:
    return int(datetime.strptime(date_text, "%Y-%m-%d")
               .replace(tzinfo=timezone.utc).timestamp() * 1000)


def fetch_oi(api_key, symbol, interval, start_ms):
    """從現在往回分頁抓 OI,直到回溯到 start_ms 或抓不到更舊的為止。"""
    headers = {"accept": "application/json", "CG-API-KEY": api_key}
    collected = {}
    end_time = None

    for _ in range(MAX_PAGES):
        params = {"symbol": symbol, "interval": interval,
                  "exchange": "Binance", "unit": "usd", "limit": PAGE_LIMIT}
        if end_time is not None:
            params["end_time"] = end_time
        resp = requests.get(config.COINGLASS_BASE_URL + OI_HISTORY_ENDPOINT,
                            headers=headers, params=params, timeout=25)
        if resp.status_code == 429:                  # 超速 -> 退避重試
            time.sleep(60)
            continue

        payload = resp.json()
        if not (isinstance(payload, dict) and str(payload.get("code")) == "0"):
            print(f"  API 回應非成功:{str(payload)[:240]}")
            break
        data = payload.get("data", [])
        if not data:
            break

        for row in data:
            collected[row["time"]] = row
        oldest = min(row["time"] for row in data)
        print(f"  已回溯到 {datetime.fromtimestamp(oldest/1000, tz=timezone.utc):%Y-%m-%d}  "
              f"(累計 {len(collected)} 筆)")

        if oldest <= start_ms or len(data) < PAGE_LIMIT:
            break
        end_time = oldest - 1
        time.sleep(SLEEP_SECONDS)

    if not collected:
        return pd.DataFrame()           # 沒抓到任何資料,回空表(由 main 處理)

    rows = [collected[t] for t in sorted(collected) if t >= start_ms]
    df = pd.DataFrame(rows)
    df["oi_time"] = pd.to_datetime(df["time"], unit="ms")
    for col in ["open", "high", "low", "close"]:
        df[f"oi_{col}"] = df[col].astype(float)
    return df[["oi_time", "oi_open", "oi_high", "oi_low", "oi_close"]] \
             .sort_values("oi_time").reset_index(drop=True)


def main():
    api_key = load_api_key()
    if not api_key:
        print("找不到 Coinglass 金鑰。請把金鑰貼到 secrets/coinglass_api_key.txt 後再跑。")
        sys.exit(1)

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    start_ms = date_to_ms(config.HISTORY_START_DATE)

    print(f"用 Coinglass 抓 {config.SYMBOL} {config.OI_INTERVAL} 未平倉量,目標回溯到 {config.HISTORY_START_DATE}")
    oi = fetch_oi(api_key, config.SYMBOL, config.OI_INTERVAL, start_ms)
    if oi.empty:
        print("沒抓到 OI 資料(看上面的 API 訊息)。")
        sys.exit(1)

    oi.to_csv(config.RAW_OI_PATH, index=False)
    print(f"\nOI {len(oi)} 筆  ({oi['oi_time'].iloc[0]} ~ {oi['oi_time'].iloc[-1]})")
    if oi["oi_time"].iloc[0] > pd.Timestamp(config.HISTORY_START_DATE):
        print(f"注意:你的方案只回溯到 {oi['oi_time'].iloc[0]:%Y-%m-%d},"
              f"沒到 {config.HISTORY_START_DATE};OI 研究期間會比 OHLCV/funding 短。")
    print(f"已存:{config.RAW_OI_PATH}")


if __name__ == "__main__":
    main()
