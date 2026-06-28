# -*- coding: utf-8 -*-
"""
event_study.py — 任務 1c:檢定「這些特徵在事件前是不是真的不一樣」。

把每根分成兩組:「未來 48h 內有 ±3% 移動(event=1)」vs「沒有(event=0)」。
對每個特徵做:
  - Mann-Whitney U 檢定:兩組的分布有沒有顯著差異(p 值越小越顯著)。
  - AUC:這個特徵單獨拿來分辨「會不會發生事件」的能力(0.5 = 跟丟銅板一樣沒用,
    越接近 1 越能分辨;< 0.5 代表反向關係)。
  - 兩組的中位數,直觀看差在哪。

重要誠實提醒:相鄰的 48h 視窗高度重疊 → 樣本不獨立 → p 值會過度樂觀。
所以同時跑一份「每 48h 取一筆」的非重疊版本當對照,後者的 p 值才比較可信。

執行:
    set PYTHONUTF8=1
    C:\\Miniconda3\\envs\\btc_lstm\\python.exe src\\event_study.py
"""

import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import config

FEATURES = ["funding_rate", "vol_24h", "absret_24h"]


def test_one_feature(df, feature):
    """對單一特徵做事件 vs 非事件的檢定,回傳一列結果。"""
    sub = df[[feature, "event"]].dropna()
    event_values = sub[sub["event"] == 1][feature]
    nonevent_values = sub[sub["event"] == 0][feature]

    _, p_value = mannwhitneyu(event_values, nonevent_values, alternative="two-sided")
    auc = roc_auc_score(sub["event"], sub[feature])

    return {
        "feature": feature,
        "median_event": event_values.median(),
        "median_nonevent": nonevent_values.median(),
        "auc": round(auc, 3),
        "p_value": p_value,
    }


def print_table(title, df):
    rows = [test_one_feature(df, f) for f in FEATURES]
    print(f"\n{title}(n={len(df.dropna(subset=['event']))})")
    print(f"  {'特徵':<12}{'事件組中位數':>14}{'非事件中位數':>14}{'AUC':>8}{'p 值':>12}")
    for r in rows:
        sig = "顯著" if r["p_value"] < 0.05 else "不顯著"
        print(f"  {r['feature']:<12}{r['median_event']:>14.6f}{r['median_nonevent']:>14.6f}"
              f"{r['auc']:>8}{r['p_value']:>12.2e}  {sig}")
    return rows


def main():
    df = pd.read_csv(config.FEATURE_TABLE_PATH, parse_dates=["open_time"])
    df = df.dropna(subset=["event"]).reset_index(drop=True)

    base_rate = df["event"].mean()
    print(f"樣本 {len(df)} 根;±{config.MOVE_THRESHOLD*100:.0f}% / {config.HORIZON_HOURS}h "
          f"事件發生率 = {base_rate*100:.1f}%")
    print("AUC 解讀:>0.5 表示特徵越大越容易發生事件;=0.5 沒鑑別力;<0.5 反向。")

    # 1) 全樣本(視窗重疊,p 值偏樂觀)
    print_table("[全樣本 — 視窗重疊,p 值偏樂觀]", df)

    # 2) 非重疊抽樣(每 HORIZON 取一筆),p 值較可信
    nonoverlap = df.iloc[::config.HORIZON_HOURS].reset_index(drop=True)
    rows = print_table(f"[非重疊抽樣 — 每 {config.HORIZON_HOURS}h 取一筆,p 值較可信]", nonoverlap)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(config.EVENT_STUDY_PATH, index=False)
    print(f"\n(非重疊版結果已存 {config.EVENT_STUDY_PATH})")


if __name__ == "__main__":
    main()
