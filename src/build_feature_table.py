# -*- coding: utf-8 -*-
"""
build_feature_table.py — 任務 1b:把 1h K 線 + 資金費率合成一張表,並標好「事件」。

做三件事:
1. 把每 8 小時的資金費率「往後填」到每根 1h(只用已結算、時間 <= 當下的,不偷看未來)。
2. 從 OHLCV 算幾個「當下已知」的特徵:近期報酬、近期波動度、近 24h 漲跌幅。
3. 標事件:未來 48 小時內,價格相對現在收盤是否出現 ±3% 的移動(event = 1 / 0)。

輸出一張 feature_table.csv,給 event_study.py 做統計檢定。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\build_feature_table.py
"""

import numpy as np
import pandas as pd

import config


def label_future_move_events(df, threshold, horizon):
    """標每一根:未來 horizon 根內,(相對現在收盤)有沒有出現 ±threshold 的移動。

    用「未來那段的最高/最低」對比現在收盤:漲幅或跌幅任一超過門檻 = 事件。
    最後 horizon 根沒有完整未來窗,標成 NaN(之後丟掉)。
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(df)
    event = np.full(n, np.nan)
    for i in range(n - horizon):
        future_high = high[i + 1:i + 1 + horizon].max()
        future_low = low[i + 1:i + 1 + horizon].min()
        up_move = future_high / close[i] - 1.0
        down_move = 1.0 - future_low / close[i]
        event[i] = 1.0 if (up_move >= threshold or down_move >= threshold) else 0.0
    return event


def main():
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    klines = pd.read_csv(config.RAW_KLINES_PATH, parse_dates=["open_time"]).sort_values("open_time")
    funding = pd.read_csv(config.RAW_FUNDING_PATH, parse_dates=["funding_time"]).sort_values("funding_time")

    # merge_asof(backward):每根 1h 取「最近一筆、時間 <= open_time」的資金費率。
    # 這保證只用「當下已經結算、已知」的費率,不會用到未來的。
    df = pd.merge_asof(klines, funding, left_on="open_time", right_on="funding_time",
                       direction="backward")

    # OHLCV 算得出來、當下已知的特徵
    returns = df["close"].pct_change()
    df["ret_1h"] = returns                                   # 上一小時報酬
    df["vol_24h"] = returns.rolling(24).std()                # 近 24h 波動度
    df["absret_24h"] = (df["close"] / df["close"].shift(24) - 1.0).abs()   # 近 24h 漲跌幅(絕對值)
    df["vol_surge"] = df["volume"] / df["volume"].rolling(24).mean()        # 成交量相對近 24h 均量(暴增 > 1)

    # 若有 OI 檔(Coinglass,可能只有近幾個月),併進來並算「近 24h 未平倉量變化」。
    # merge_asof backward:每根 1h 取最近一筆、時間 <= 當下的 OI,不偷看未來。
    if config.RAW_OI_PATH.exists():
        oi = pd.read_csv(config.RAW_OI_PATH, parse_dates=["oi_time"]).sort_values("oi_time")
        df = pd.merge_asof(df, oi[["oi_time", "oi_close"]], left_on="open_time",
                           right_on="oi_time", direction="backward")
        df["oi_change_24h"] = df["oi_close"] / df["oi_close"].shift(24) - 1.0
        print(f"已併入 OI:{len(oi)} 筆(自 {oi['oi_time'].iloc[0]:%Y-%m-%d});"
              f"有 OI 的根數 {df['oi_close'].notna().sum()}")
    else:
        print("(沒有 OI 檔,略過 OI 特徵)")

    # 事件標籤:未來 48h 內 ±3%
    df["event"] = label_future_move_events(df, config.MOVE_THRESHOLD, config.HORIZON_HOURS)

    df.to_csv(config.FEATURE_TABLE_PATH, index=False)

    valid = df.dropna(subset=["event"])
    print(f"特徵表 {len(df)} 根;可算事件的 {len(valid)} 根")
    print(f"±{config.MOVE_THRESHOLD*100:.0f}% / {config.HORIZON_HOURS}h 事件發生率:{valid['event'].mean()*100:.1f}%")
    print(f"欄位:{list(df.columns)}")
    print(f"已存:{config.FEATURE_TABLE_PATH}")


if __name__ == "__main__":
    main()
