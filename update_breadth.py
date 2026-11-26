import json
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd
import yfinance as yf

# 範例股票池（日後可擴充為完整港股通或自選股清單）
TICKERS = [
    "0700.HK", "9988.HK", "3690.HK", "1810.HK", "0981.HK",
    "1211.HK", "9618.HK", "0020.HK", "2318.HK", "0388.HK",
    "9888.HK", "9999.HK", "2015.HK", "9868.HK", "9866.HK",
    "2382.HK", "0992.HK", "0285.HK", "1024.HK", "0763.HK",
    "0005.HK", "1299.HK", "0941.HK", "0883.HK", "0939.HK",
    "1398.HK", "3968.HK", "0669.HK", "2269.HK", "1833.HK",
]


def calculate_market_monitor(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    min_price: float = 1.0,
    min_volume: int = 100_000,
    min_turnover: float = 5_000_000,
    require_vol_expansion: bool = True,
) -> pd.DataFrame:
    close_df = close_df.sort_index().dropna(how="all")
    volume_df = volume_df.reindex_like(close_df)

    # 流動性過濾遮罩
    turnover_df = close_df * volume_df
    liquid_mask = (
        close_df.notna()
        & (close_df >= min_price)
        & (volume_df >= min_volume)
        & (turnover_df >= min_turnover)
    )

    # 1 日與 25 日漲跌幅（fill_method=None 避免停牌股被誤填為 0%）
    ret_1d = close_df.pct_change(1, fill_method=None)
    ret_25d = close_df.pct_change(25, fill_method=None)

    # 成交量需高於前一日（可按需要開關）
    if require_vol_expansion:
        vol_cond = volume_df > volume_df.shift(1)
    else:
        vol_cond = pd.DataFrame(True, index=close_df.index, columns=close_df.columns)

    # 單日升跌 4% 家數
    up_4pct = ((ret_1d >= 0.04) & liquid_mask & vol_cond).sum(axis=1)
    down_4pct = ((ret_1d <= -0.04) & liquid_mask & vol_cond).sum(axis=1)

    # 5 日動能比率
    up_4pct_5d = up_4pct.rolling(5, min_periods=5).sum()
    down_4pct_5d = down_4pct.rolling(5, min_periods=5).sum()
    ratio_5d = (up_4pct_5d / down_4pct_5d.replace(0, np.nan)).round(2)

    # 25 日強弱廣度（升跌 25% 以上）
    up_25pct_25d = ((ret_25d >= 0.25) & liquid_mask).sum(axis=1)
    down_25pct_25d = ((ret_25d <= -0.25) & liquid_mask).sum(axis=1)

    # 活躍股數與跌 4% 佔比
    active_stocks = liquid_mask.sum(axis=1)
    down_4pct_pct = (down_4pct / active_stocks.replace(0, np.nan) * 100).round(2)

    # 站上 40 日均線比例
    sma40 = close_df.rolling(40, min_periods=40).mean()
    above_40sma = ((close_df > sma40) & liquid_mask).sum(axis=1)
    pct_above_40sma = (above_40sma / active_stocks.replace(0, np.nan) * 100).round(1)

    breadth_df = pd.DataFrame(
        {
            "Active_Stocks": active_stocks.astype(int),
            "Up_4pct": up_4pct.astype(int),
            "Down_4pct": down_4pct.astype(int),
            "Down_4pct_Pct": down_4pct_pct,
            "Ratio_5D": ratio_5d,
            "Up_25pct_25D": up_25pct_25d.astype(int),
            "Down_25pct_25D": down_25pct_25d.astype(int),
            "Pct_Above_40SMA": pct_above_40sma,
        },
        index=close_df.index,
    )

    # 判斷市況狀態
    conditions = [
        (breadth_df["Ratio_5D"] >= 2.0) & (breadth_df["Up_25pct_25D"] > breadth_df["Down_25pct_25D"]),
        (breadth_df["Ratio_5D"] < 1.0) | (breadth_df["Down_4pct_Pct"] >= 8.0),
    ]
    choices = ["Bullish", "Correction"]
    breadth_df["Regime"] = np.select(conditions, choices, default="Choppy")
    breadth_df.index.name = "Date"
    return breadth_df


def main():
    raw = yf.download(
        TICKERS,
        period="9mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    close_df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    volume_df = raw["Volume"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Volume"]]

    breadth_df = calculate_market_monitor(close_df, volume_df)

    # 保留最近 90 個交易日供網頁快速載入
    recent = breadth_df.tail(90).reset_index()
    recent["Date"] = pd.to_datetime(recent["Date"]).dt.strftime("%Y-%m-%d")
    recent = recent.replace([np.inf, -np.inf], np.nan)

    # 將 NaN 轉為 None，確保 JSON 格式符合瀏覽器標準
    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in recent.to_dict(orient="records")
    ]

    hkt = timezone(timedelta(hours=8))
    output = {
        "updated_at": datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT"),
        "universe_size": len(TICKERS),
        "history": records,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"已成功輸出 data.json，共 {len(records)} 個交易日。")


if __name__ == "__main__":
    main()
