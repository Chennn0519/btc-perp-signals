# -*- coding: utf-8 -*-
"""
event_study.py — 任務 1c:檢定「這些特徵在事件前是不是真的不一樣」。

把每根分成兩組:「未來 48h 內有 ±3% 移動(event=1)」vs「沒有(event=0)」。
對每個特徵做:
  - Mann-Whitney U 檢定:兩組分布有沒有顯著差異(p 越小越顯著)。
  - AUC:這個特徵單獨分辨「會不會發生事件」的能力(0.5 = 沒鑑別力,越接近 1 越能分辨;
    < 0.5 代表反向)。
  - 兩組中位數;以及各特徵的有效樣本數 n(OI 只有近幾個月,n 會明顯較少)。

誠實提醒:相鄰 48h 視窗重疊 → 樣本不獨立 → p 值偏樂觀。故同時跑「每 48h 取一筆」的
非重疊版本,後者 p 值較可信。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\event_study.py
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import config

BASE_FEATURES = ["funding_rate", "vol_24h", "absret_24h"]


def test_one_feature(df, feature):
    sub = df[[feature, "event"]].dropna()
    event_values = sub[sub["event"] == 1][feature]
    nonevent_values = sub[sub["event"] == 0][feature]
    _, p_value = mannwhitneyu(event_values, nonevent_values, alternative="two-sided")
    auc = roc_auc_score(sub["event"], sub[feature])
    return {"feature": feature, "n": len(sub),
            "median_event": event_values.median(),
            "median_nonevent": nonevent_values.median(),
            "auc": round(auc, 3), "p_value": p_value}


def print_table(title, df, features):
    print(f"\n{title}")
    print(f"  {'特徵':<14}{'n':>7}{'事件組中位數':>14}{'非事件中位數':>14}{'AUC':>7}{'p 值':>11}")
    rows = []
    for feature in features:
        r = test_one_feature(df, feature)
        rows.append(r)
        sig = "顯著" if r["p_value"] < 0.05 else "不顯著"
        print(f"  {r['feature']:<14}{r['n']:>7}{r['median_event']:>14.6f}"
              f"{r['median_nonevent']:>14.6f}{r['auc']:>7}{r['p_value']:>11.1e}  {sig}")
    return rows


def main():
    df = pd.read_csv(config.FEATURE_TABLE_PATH, parse_dates=["open_time"])
    df = df.dropna(subset=["event"]).reset_index(drop=True)

    features = list(BASE_FEATURES)
    if "oi_change_24h" in df.columns:
        features.append("oi_change_24h")

    base_rate = df["event"].mean()
    print(f"樣本 {len(df)} 根;±{config.MOVE_THRESHOLD*100:.0f}% / {config.HORIZON_HOURS}h "
          f"事件發生率 = {base_rate*100:.1f}%")
    print("AUC 解讀:>0.5 表示特徵越大越容易發生事件;=0.5 沒鑑別力;<0.5 反向。")

    print_table("[全樣本 — 視窗重疊,p 值偏樂觀]", df, features)

    nonoverlap = df.iloc[::config.HORIZON_HOURS].reset_index(drop=True)
    rows = print_table(f"[非重疊抽樣 — 每 {config.HORIZON_HOURS}h 取一筆,p 值較可信]",
                       nonoverlap, features)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.EVENT_STUDY_PATH, index=False)
    print(f"\n(非重疊版結果已存 {config.EVENT_STUDY_PATH})")


if __name__ == "__main__":
    main()
