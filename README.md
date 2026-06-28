# btc-perp-signals

BTC 永續合約「訊號研究」。目標不是訓練模型預測價格,而是先用**統計檢定**驗證:
**某些衍生品特徵(資金費率、未平倉量…),在大行情發生前是不是真的不一樣?**
也就是先找「能被證明有效的訊號來源」,有訊號再談模型。

(這個方向來自 schumi/Gemini 的建議:「你們不缺交易機器人,缺的是能被證明有效的
訊號來源。」與 btc-15m-lstm 的結論一致——那個專案證明了 15 分鐘方向用 LSTM 測不準,
瓶頸不在模型,在訊號。)

## 與 btc-15m-lstm 的差別

| | btc-15m-lstm | btc-perp-signals(本專案) |
| --- | --- | --- |
| 資料 | 現貨 15 分 | 永續合約 1 小時 + 衍生品 |
| 方法 | LSTM 預測價格 | 統計檢定找訊號 |
| 問題 | 下一根多少錢 | 哪些資料在大行情前不一樣 |

## 環境

沿用 conda 環境 `btc_lstm`(已含 pandas、numpy、scipy、scikit-learn、requests)。

```
set PYTHONUTF8=1
C:\Miniconda3\envs\btc_lstm\python.exe src\download_perp_data.py
```

## 第一任務的流程

```
download_perp_data → build_feature_table → event_study
  抓 1h OHLCV+funding    合表+標事件         統計檢定
```

- `src/config.py` — 設定集中地(API、幣別、事件定義 ±3%/48h)。
- `src/download_perp_data.py` — 抓永續 1h K 線 + 資金費率(都有完整歷史)。
- `src/build_feature_table.py` — 合成一張表;資金費率往後填到 1h(不偷看未來);
  算近期波動/漲跌特徵;標「未來 48h 內 ±3%」事件。
- `src/event_study.py` — 檢定每個特徵在「事件 vs 非事件」兩組是否有顯著差異
  (Mann-Whitney + AUC),並附「非重疊抽樣」版以免 p 值過度樂觀。

## 資料現況與限制

- 1h OHLCV(永續):Binance 完整歷史 ✓
- 資金費率:Binance 完整歷史 ✓(每 8 小時一筆)
- 未平倉量(OI):**Binance 免費只給最近約 30 天**,做 2.5 年研究不夠;長歷史 1h OI
  需用 Coinglass 訂閱抓(待辦)。
- 後續還要補:Liquidation、Order Book、BTC/ETH/美股關聯。新聞與鏈上先不碰。
