# -*- coding: utf-8 -*-
"""
signal_lab.py — 訊號實驗室:產出 schumi 要的里程碑表。

問題:把某個特徵分成「高分組(前 25%)」和「低分組(後 25%)」,
兩組「未來 48h 內出現 ±3% 單邊行情」的發生率差多少?有沒有穩定差異?

只要有一個特徵的高分組事件率明顯 > 低分組,就代表這個資料源「有預測力」,
才有資格進入模型階段。

測的特徵:
  - Funding 極端值 = |funding_rate|(資金費率多空都算極端)
  - OI 暴增       = oi_change_24h(近 24h 未平倉量變化;OI 只有近幾個月)
  - 成交量暴增     = vol_surge(成交量 / 近 24h 均量)
  - 近期波動       = vol_24h(對照組,前面已知是有效訊號)

誠實提醒:相鄰 48h 視窗重疊 → 顯著性會被高估。故同時跑「每 48h 取一筆」的非重疊版,
後者才可信。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\signal_lab.py
"""

import pandas as pd
from scipy.stats import chi2_contingency

import config

TOP_FRAC = 0.25   # 高分組 = 特徵前 25%、低分組 = 後 25%

# (顯示名稱, 欄位, 取值方式:abs 取絕對值 / raw 原值)
SIGNALS = [
    ("Funding 極端值", "funding_rate", "abs"),
    ("OI 暴增", "oi_change_24h", "raw"),
    ("成交量暴增", "vol_surge", "raw"),
    ("近期波動", "vol_24h", "raw"),
]


def feature_values(df, column, mode):
    return df[column].abs() if mode == "abs" else df[column]


def high_low_event_rates(feature, event, top_frac):
    """把特徵分高/低分組,回傳兩組事件率、差異、卡方 p 值。"""
    s = pd.DataFrame({"f": feature, "e": event}).dropna()
    if len(s) < 50:
        return None
    high = s[s["f"] >= s["f"].quantile(1 - top_frac)]
    low = s[s["f"] <= s["f"].quantile(top_frac)]
    table = [[(high["e"] == 1).sum(), (high["e"] == 0).sum()],
             [(low["e"] == 1).sum(), (low["e"] == 0).sum()]]
    _, p_value, _, _ = chi2_contingency(table)
    return {"hi_rate": high["e"].mean(), "lo_rate": low["e"].mean(),
            "diff": high["e"].mean() - low["e"].mean(),
            "p_value": p_value, "n_hi": len(high), "n_lo": len(low)}


def print_table(title, df):
    print(f"\n{title}")
    print(f"  {'特徵':<14}{'高分組事件率':>13}{'低分組事件率':>13}{'差異':>9}{'p 值':>10}  判讀")
    for name, column, mode in SIGNALS:
        if column not in df.columns:
            continue
        r = high_low_event_rates(feature_values(df, column, mode), df["event"], TOP_FRAC)
        if r is None:
            print(f"  {name:<14}  資料不足")
            continue
        verdict = "有差異" if r["p_value"] < 0.05 else "無顯著差異"
        print(f"  {name:<14}{r['hi_rate']*100:>12.1f}%{r['lo_rate']*100:>12.1f}%"
              f"{r['diff']*100:>+8.1f}{r['p_value']:>10.1e}  {verdict}"
              f"  (n 高{r['n_hi']}/低{r['n_lo']})")


def main():
    df = pd.read_csv(config.FEATURE_TABLE_PATH, parse_dates=["open_time"])
    df = df.dropna(subset=["event"]).reset_index(drop=True)

    print(f"事件 = 未來 {config.HORIZON_HOURS}h 內 ±{config.MOVE_THRESHOLD*100:.0f}% 單邊行情;"
          f"整體發生率 {df['event'].mean()*100:.1f}%(高/低分組要明顯偏離這個才算有訊號)")

    print_table("[全樣本 — 視窗重疊,顯著性偏樂觀]", df)

    nonoverlap = df.iloc[::config.HORIZON_HOURS].reset_index(drop=True)
    print_table(f"[非重疊抽樣 — 每 {config.HORIZON_HOURS}h 取一筆,較可信]", nonoverlap)


if __name__ == "__main__":
    main()
