import json
import os
import random
import time
from datetime import datetime, timezone, timedelta

import akshare as ak
import numpy as np
import pandas as pd
import yfinance as yf

UNIVERSE_CACHE_FILE = "ggt_universe.json"

# 保底清單（僅在首次執行且無法連線抓取名單時啟用）
FALLBACK_STOCKS = {
    "0700.HK": "騰訊控股", "9988.HK": "阿里巴巴-W", "3690.HK": "美團-W",
    "1810.HK": "小米集團-W", "0981.HK": "中芯國際", "1211.HK": "比亞迪股份",
    "9618.HK": "京東集團-SW", "0020.HK": "商湯-W", "2318.HK": "中國平安",
    "0388.HK": "香港交易所", "9888.HK": "百度集團-SW", "9999.HK": "網易-S",
    "2015.HK": "理想汽車-W", "9868.HK": "小鵬汽車-W", "9866.HK": "蔚來-SW",
    "2382.HK": "舜宇光學科技", "0992.HK": "聯想集團", "0285.HK": "比亞迪電子",
    "1024.HK": "快手-W", "0763.HK": "中興通訊", "0005.HK": "匯豐控股",
    "1299.HK": "友邦保險", "0941.HK": "中國移動", "0883.HK": "中國海洋石油",
    "0939.HK": "建設銀行", "1398.HK": "工商銀行", "3968.HK": "招商銀行",
    "0669.HK": "創科實業", "2269.HK": "藥明生物", "1833.HK": "平安好醫生",
}


def safe_round(val, digits=2):
    if val is None or pd.isna(val) or np.isinf(val):
        return None
    return round(float(val), digits)


def normalize_hk_ticker(raw_code: str) -> str | None:
    """將 5 位數港股代碼轉為 Yahoo Finance 格式（如 00700 -> 0700.HK），排除 8xxxx 人民幣櫃台。"""
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if not digits:
        return None
    num = int(digits)
    if 1 <= num <= 9999:
        return f"{num:04d}.HK"
    return None


def fetch_stock_connect_universe(max_retries: int = 3) -> dict[str, str]:
    """三層容錯獲取港股通完整名單：AkShare 即時抓取 -> 本地快取 -> 保底清單。"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[名單同步] 第 {attempt}/{max_retries} 次透過 AkShare 抓取港股通成分股...")
            df = ak.stock_hk_ggt_components_em()
            if df is None or df.empty:
                raise ValueError("回傳資料表為空")

            code_col = next((c for c in df.columns if "代码" in c or "代碼" in c), None)
            name_col = next((c for c in df.columns if "名称" in c or "名稱" in c), None)
            if not code_col or not name_col:
                raise ValueError(f"找不到代碼或名稱欄位：{list(df.columns)}")

            stock_map = {}
            for _, row in df.iterrows():
                yf_ticker = normalize_hk_ticker(row[code_col])
                if yf_ticker:
                    stock_map[yf_ticker] = str(row[name_col]).strip()

            if len(stock_map) >= 300:
                with open(UNIVERSE_CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump(stock_map, f, ensure_ascii=False, indent=2)
                print(f"[名單同步] 成功取得 {len(stock_map)} 隻港股通標的，已更新 {UNIVERSE_CACHE_FILE}")
                return stock_map

            raise ValueError(f"取得家數異常偏少（僅 {len(stock_map)} 隻），疑似資料截斷")
        except Exception as exc:
            print(f"[名單同步] 第 {attempt} 次失敗：{exc}")
            if attempt < max_retries:
                time.sleep(3 * attempt)

    if os.path.exists(UNIVERSE_CACHE_FILE):
        try:
            with open(UNIVERSE_CACHE_FILE, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if isinstance(cached, dict) and len(cached) > 0:
                print(f"[名單同步] 改用本地快取 {UNIVERSE_CACHE_FILE}（共 {len(cached)} 隻）")
                return cached
        except Exception as exc:
            print(f"[名單同步] 讀取快取檔失敗：{exc}")

    print(f"[名單同步] 啟用保底清單（共 {len(FALLBACK_STOCKS)} 隻）")
    return FALLBACK_STOCKS.copy()


def extract_ohlcv_from_batch(raw: pd.DataFrame, batch: list[str]) -> dict[str, pd.DataFrame]:
    """相容不同版本 yfinance 與單一股票批次的欄位提取器。"""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    extracted = {}
    if raw is None or raw.empty:
        return extracted

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        for field in fields:
            if field in level0:
                sub = raw[field]
            elif field in level1:
                sub = raw.xs(field, axis=1, level=1)
            else:
                continue
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(name=batch[0])
            extracted[field] = sub
    else:
        for field in fields:
            if field in raw.columns:
                extracted[field] = raw[[field]].rename(columns={field: batch[0]})

    return extracted


def download_ohlcv_in_batches(
    tickers: list[str],
    period: str = "9mo",
    batch_size: int = 50,
    max_retries: int = 3,
) -> dict[str, pd.DataFrame]:
    """分批向 Yahoo Finance 下載日線 OHLCV，含隨機延遲與失敗重試。"""
    fields = ["Open", "High", "Low", "Close", "Volume"]
    collected = {f: [] for f in fields}

    total_batches = (len(tickers) + batch_size - 1) // batch_size
    failed_batches = []

    for idx in range(0, len(tickers), batch_size):
        batch = tickers[idx : idx + batch_size]
        batch_no = idx // batch_size + 1
        success = False

        for attempt in range(1, max_retries + 1):
            try:
                raw = yf.download(
                    batch,
                    period=period,
                    interval="1d",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=30,
                )
                parsed = extract_ohlcv_from_batch(raw, batch)
                if "Close" not in parsed or parsed["Close"].dropna(how="all").empty:
                    raise ValueError("下載結果缺少有效收盤價")

                for f in fields:
                    if f in parsed:
                        collected[f].append(parsed[f])

                print(f"[行情下載] 批次 {batch_no}/{total_batches} 成功（{len(batch)} 隻）")
                success = True
                break
            except Exception as exc:
                wait_sec = 5 * attempt + random.uniform(1.0, 3.0)
                print(
                    f"[行情下載] 批次 {batch_no}/{total_batches} 第 {attempt} 次失敗：{exc}"
                    f" -> {wait_sec:.1f} 秒後重試"
                )
                if attempt < max_retries:
                    time.sleep(wait_sec)

        if not success:
            failed_batches.append(batch_no)

        if batch_no < total_batches:
            time.sleep(random.uniform(1.5, 3.0))

    if failed_batches:
        print(f"[行情下載] 警告：以下批次最終失敗並略過：{failed_batches}")

    merged = {}
    for f in fields:
        if not collected[f]:
            raise RuntimeError(f"未能成功下載任何 {f} 行情數據")
        df = pd.concat(collected[f], axis=1)
        df = df.loc[:, ~df.columns.duplicated()].sort_index()
        merged[f] = df.dropna(how="all")

    print(f"[行情下載] 合併完成，共取得 {merged['Close'].shape[1]} 隻有效股票行情。")
    return merged


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

    turnover_df = close_df * volume_df
    liquid_mask = (
        close_df.notna()
        & (close_df >= min_price)
        & (volume_df >= min_volume)
        & (turnover_df >= min_turnover)
    )

    ret_1d = close_df.pct_change(1, fill_method=None)
    ret_25d = close_df.pct_change(25, fill_method=None)

    if require_vol_expansion:
        vol_cond = volume_df > volume_df.shift(1)
    else:
        vol_cond = pd.DataFrame(True, index=close_df.index, columns=close_df.columns)

    up_4pct = ((ret_1d >= 0.04) & liquid_mask & vol_cond).sum(axis=1)
    down_4pct = ((ret_1d <= -0.04) & liquid_mask & vol_cond).sum(axis=1)

    up_4pct_5d = up_4pct.rolling(5, min_periods=5).sum()
    down_4pct_5d = down_4pct.rolling(5, min_periods=5).sum()
    ratio_5d = (up_4pct_5d / down_4pct_5d.replace(0, np.nan)).round(2)

    up_25pct_25d = ((ret_25d >= 0.25) & liquid_mask).sum(axis=1)
    down_25pct_25d = ((ret_25d <= -0.25) & liquid_mask).sum(axis=1)

    active_stocks = liquid_mask.sum(axis=1)
    up_4pct_pct = (up_4pct / active_stocks.replace(0, np.nan) * 100).round(2)
    down_4pct_pct = (down_4pct / active_stocks.replace(0, np.nan) * 100).round(2)

    sma40 = close_df.rolling(40, min_periods=40).mean()
    above_40sma = ((close_df > sma40) & liquid_mask).sum(axis=1)
    pct_above_40sma = (above_40sma / active_stocks.replace(0, np.nan) * 100).round(1)

    breadth_df = pd.DataFrame(
        {
            "Active_Stocks": active_stocks.astype(int),
            "Up_4pct": up_4pct.astype(int),
            "Down_4pct": down_4pct.astype(int),
            "Up_4pct_Pct": up_4pct_pct,
            "Down_4pct_Pct": down_4pct_pct,
            "Ratio_5D": ratio_5d,
            "Up_25pct_25D": up_25pct_25d.astype(int),
            "Down_25pct_25D": down_25pct_25d.astype(int),
            "Pct_Above_40SMA": pct_above_40sma,
        },
        index=close_df.index,
    )

    median_active = breadth_df["Active_Stocks"].tail(30).median()
    if pd.notna(median_active) and median_active > 0:
        breadth_df = breadth_df[breadth_df["Active_Stocks"] >= median_active * 0.3]

    conditions = [
        (breadth_df["Ratio_5D"] >= 2.0) & (breadth_df["Up_25pct_25D"] > breadth_df["Down_25pct_25D"]),
        (breadth_df["Ratio_5D"] < 1.0) | (breadth_df["Down_4pct_Pct"] >= 8.0),
    ]
    choices = ["Bullish", "Correction"]
    breadth_df["Regime"] = np.select(conditions, choices, default="Choppy")
    breadth_df.index.name = "Date"
    return breadth_df


def scan_daily_candidates(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    stock_names: dict[str, str],
    min_price: float = 1.0,
    min_volume: int = 100_000,
    min_turnover: float = 5_000_000,
) -> dict:
    """掃描最新一個交易日符合『放量突破』與『超跌反轉』條件的候選股。"""
    close_df = close_df.sort_index().dropna(how="all")
    open_df = open_df.reindex_like(close_df)
    high_df = high_df.reindex_like(close_df)
    low_df = low_df.reindex_like(close_df)
    volume_df = volume_df.reindex_like(close_df)

    if len(close_df) < 55:
        return {"breakouts": [], "reversals": []}

    sma20 = close_df.rolling(20, min_periods=20).mean()
    sma50 = close_df.rolling(50, min_periods=50).mean()
    vol_sma20_prev = volume_df.shift(1).rolling(20, min_periods=20).mean()
    high_20d_prev = high_df.shift(1).rolling(20, min_periods=20).max()
    high_15d_prev = high_df.shift(1).rolling(15, min_periods=15).max()

    o_t, h_t, l_t, c_t, v_t = (
        open_df.iloc[-1], high_df.iloc[-1], low_df.iloc[-1], close_df.iloc[-1], volume_df.iloc[-1]
    )
    c_prev, v_prev = close_df.iloc[-2], volume_df.iloc[-2]
    sma20_t, sma20_prev, sma50_t = sma20.iloc[-1], sma20.iloc[-2], sma50.iloc[-1]
    vol_avg_t = vol_sma20_prev.iloc[-1]
    h20_prev = high_20d_prev.iloc[-1]
    h15_prev = high_15d_prev.iloc[-1]

    breakouts = []
    reversals = []

    for ticker in close_df.columns:
        c = c_t.get(ticker)
        o = o_t.get(ticker)
        h = h_t.get(ticker)
        l = l_t.get(ticker)
        v = v_t.get(ticker)
        cp = c_prev.get(ticker)
        vp = v_prev.get(ticker)
        va = vol_avg_t.get(ticker)

        if any(pd.isna(x) for x in [c, o, h, l, v, cp, vp, va]):
            continue
        if c < min_price or v < min_volume or (c * v) < min_turnover or cp <= 0 or va <= 0:
            continue

        ret_1d = (c / cp) - 1.0
        vol_ratio = v / va
        day_range = h - l
        close_pos = (c - l) / day_range if day_range > 0 else 0.5
        s20 = sma20_t.get(ticker)
        s20_p = sma20_prev.get(ticker)
        s50 = sma50_t.get(ticker)
        h20 = h20_prev.get(ticker)
        h15 = h15_prev.get(ticker)

        # 1. 放量突破條件 (Breakout)
        is_breakout = (
            pd.notna(s20) and pd.notna(s50) and pd.notna(h20)
            and c > s20 and c > s50
            and c >= h20
            and ret_1d >= 0.04
            and v > vp and vol_ratio >= 1.5
            and close_pos >= 0.70
        )

        if is_breakout:
            dist_sma20 = (c / s20 - 1.0) * 100
            breakouts.append(
                {
                    "ticker": ticker,
                    "name": stock_names.get(ticker, ""),
                    "close": safe_round(c, 2),
                    "change_pct": safe_round(ret_1d * 100, 2),
                    "vol_ratio": safe_round(vol_ratio, 2),
                    "close_pos": safe_round(close_pos * 100, 0),
                    "metric_label": f"偏離20MA +{dist_sma20:.1f}%",
                }
            )
            continue

        # 2. 超跌反轉條件 (Reversal)
        drawdown_15d = (cp / h15 - 1.0) if (pd.notna(h15) and h15 > 0) else 0.0
        is_reversal = (
            pd.notna(s20_p)
            and drawdown_15d <= -0.10
            and cp < s20_p
            and ret_1d >= 0.03
            and c > o
            and close_pos >= 0.60
            and v > vp and vol_ratio >= 1.2
        )

        if is_reversal:
            reversals.append(
                {
                    "ticker": ticker,
                    "name": stock_names.get(ticker, ""),
                    "close": safe_round(c, 2),
                    "change_pct": safe_round(ret_1d * 100, 2),
                    "vol_ratio": safe_round(vol_ratio, 2),
                    "close_pos": safe_round(close_pos * 100, 0),
                    "metric_label": f"前15日回調 {drawdown_15d * 100:.1f}%",
                }
            )

    breakouts.sort(key=lambda x: x["change_pct"] or 0, reverse=True)
    reversals.sort(key=lambda x: x["change_pct"] or 0, reverse=True)

    return {
        "breakouts": breakouts[:20],
        "reversals": reversals[:20],
    }


def generate_daily_insights(
    breadth_df: pd.DataFrame,
    candidates: dict,
    universe_size: int,
    heavy_sell_pct: float = 8.0,
) -> dict:
    """產出供網頁呈現的市況轉折提示、單日變化與可複製收市文字摘要。"""
    df = breadth_df.sort_index()
    recent_5d = df.tail(5)
    latest = recent_5d.iloc[-1]
    prev = recent_5d.iloc[-2] if len(recent_5d) >= 2 else latest
    latest_date = pd.to_datetime(latest.name).strftime("%Y-%m-%d")

    regime = str(latest.get("Regime", "Choppy"))
    active = int(latest["Active_Stocks"])
    up_4 = int(latest["Up_4pct"])
    dn_4 = int(latest["Down_4pct"])
    up_4_pct = float(latest["Up_4pct_Pct"]) if pd.notna(latest["Up_4pct_Pct"]) else 0.0
    dn_4_pct = float(latest["Down_4pct_Pct"]) if pd.notna(latest["Down_4pct_Pct"]) else 0.0

    r5_now = latest["Ratio_5D"]
    r5_prev = prev["Ratio_5D"]
    r5_diff = safe_round(r5_now - r5_prev, 2) if (pd.notna(r5_now) and pd.notna(r5_prev)) else None

    up_25 = int(latest["Up_25pct_25D"])
    dn_25 = int(latest["Down_25pct_25D"])
    net_25 = up_25 - dn_25
    sma40_pct = latest["Pct_Above_40SMA"]

    # 1. 偵測關鍵轉折與異常警訊（level: danger / warning / success）
    alerts = []
    heavy_sell_days = int((recent_5d["Down_4pct_Pct"] >= heavy_sell_pct).sum())
    if heavy_sell_days >= 2:
        alerts.append(
            {
                "level": "danger",
                "text": f"近 5 日出現 {heavy_sell_days} 次廣泛拋售（跌 4%+ 佔比 ≥ {heavy_sell_pct:.0f}%）",
            }
        )

    if pd.notna(r5_now) and pd.notna(r5_prev):
        if r5_prev >= 1.0 and r5_now < 1.0:
            alerts.append({"level": "danger", "text": "5 日動能比率剛跌破 1.0，短線轉入空方主導"})
        elif r5_prev < 2.0 and r5_now >= 2.0:
            alerts.append({"level": "success", "text": "5 日動能比率重新站上 2.0，買盤動能轉強"})

    if up_25 < dn_25 and int(prev["Up_25pct_25D"]) >= int(prev["Down_25pct_25D"]):
        alerts.append({"level": "warning", "text": "25 日弱勢股家數反超強勢股，中期波段結構轉弱"})

    if pd.notna(sma40_pct) and sma40_pct <= 20.0:
        alerts.append(
            {
                "level": "warning",
                "text": f"站上 40 日均線比例跌至 {sma40_pct:.1f}%（極端超賣區，留意反轉訊號）",
            }
        )

    # 2. 組合純文字收市摘要（方便網頁檢視與一鍵複製）
    if regime == "Bullish":
        status_text = "🟢 多頭擴張（可積極追蹤突破）"
    elif regime == "Correction":
        status_text = "🔴 調整／偏空（防守為主，做多僅限反轉）"
    else:
        status_text = "🟡 震盪轉折（減倉並收緊止損）"

    if pd.notna(r5_now) and r5_diff is not None:
        arrow = "▲" if r5_diff > 0 else ("▼" if r5_diff < 0 else "▬")
        r5_str = f"{r5_now:.2f} ({arrow}{abs(r5_diff):.2f})"
    elif pd.notna(r5_now):
        r5_str = f"{r5_now:.2f}"
    else:
        r5_str = "N/A"

    sma40_str = f"{sma40_pct:.1f}%" if pd.notna(sma40_pct) else "N/A"
    net_25_str = f"+{net_25}" if net_25 > 0 else f"{net_25}"

    table_lines = ["Date   +4%  -4%  Ratio  25D(+/-)", "-" * 32]
    for dt, row in recent_5d.iterrows():
        d_str = pd.to_datetime(dt).strftime("%m-%d")
        u4 = int(row["Up_4pct"])
        d4 = int(row["Down_4pct"])
        r5 = f"{row['Ratio_5D']:5.2f}" if pd.notna(row["Ratio_5D"]) else "  N/A"
        s25 = f"{int(row['Up_25pct_25D'])}/{int(row['Down_25pct_25D'])}"
        table_lines.append(f"{d_str}  {u4:3d}  {d4:3d}  {r5}  {s25:>8}")

    def fmt_list(items, limit=8):
        if not items:
            return ["• 今日無符合條件標的"]
        out = [
            f"• {x['ticker']} {x['name']}｜${x['close']:.2f} (+{x['change_pct']:.1f}%｜量比{x['vol_ratio']:.1f}x)"
            for x in items[:limit]
        ]
        if len(items) > limit:
            out.append(f"  ...其餘 {len(items) - limit} 隻請見下方完整表格")
        return out

    breakouts = candidates.get("breakouts", [])
    reversals = candidates.get("reversals", [])

    summary_lines = [
        f"📊 港股通 Market Monitor ({latest_date})",
        f"狀態：{status_text}",
        "",
        f"📌 當日廣度快照（活躍 {active} / 總池 {universe_size} 隻）",
        f"• 單日升 4%+：🟢 {up_4} 隻 ({up_4_pct:.1f}%)",
        f"• 單日跌 4%+：🔴 {dn_4} 隻 ({dn_4_pct:.1f}%)",
        f"• 5 日動能比率：{r5_str}",
        f"• 25 日強/弱股：{up_25} vs {dn_25}（淨差 {net_25_str}）",
        f"• 站上 40 日均線：{sma40_str}",
        "",
        "📈 近 5 日廣度趨勢",
        *table_lines,
    ]

    if alerts:
        summary_lines.extend(["", "⚠️ 市況提示"])
        summary_lines.extend(f"• {a['text']}" for a in alerts)

    if regime == "Correction":
        summary_lines.extend(["", f"🎯 超跌反轉（調整期優先｜共 {len(reversals)} 隻）", *fmt_list(reversals)])
        summary_lines.extend(["", f"🚀 放量突破（調整期慎追｜共 {len(breakouts)} 隻）", *fmt_list(breakouts, 5)])
    else:
        summary_lines.extend(["", f"🚀 放量突破（共 {len(breakouts)} 隻）", *fmt_list(breakouts)])
        summary_lines.extend(["", f"🔄 超跌反轉（共 {len(reversals)} 隻）", *fmt_list(reversals, 5)])

    return {
        "ratio_5d_diff": r5_diff,
        "alerts": alerts,
        "summary_text": "\n".join(summary_lines),
    }


def main():
    start_ts = time.time()
    stock_names = fetch_stock_connect_universe(max_retries=3)
    tickers = sorted(stock_names.keys())

    ohlcv = download_ohlcv_in_batches(
        tickers=tickers,
        period="9mo",
        batch_size=50,
        max_retries=3,
    )

    breadth_df = calculate_market_monitor(ohlcv["Close"], ohlcv["Volume"])
    candidates = scan_daily_candidates(
        open_df=ohlcv["Open"],
        high_df=ohlcv["High"],
        low_df=ohlcv["Low"],
        close_df=ohlcv["Close"],
        volume_df=ohlcv["Volume"],
        stock_names=stock_names,
    )
    insights = generate_daily_insights(
        breadth_df=breadth_df,
        candidates=candidates,
        universe_size=len(tickers),
    )

    recent = breadth_df.tail(90).reset_index()
    recent["Date"] = pd.to_datetime(recent["Date"]).dt.strftime("%Y-%m-%d")
    recent = recent.replace([np.inf, -np.inf], np.nan)

    records = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in recent.to_dict(orient="records")
    ]

    hkt = timezone(timedelta(hours=8))
    output = {
        "updated_at": datetime.now(hkt).strftime("%Y-%m-%d %H:%M HKT"),
        "universe_size": len(tickers),
        "history": records,
        "candidates": candidates,
        "insights": insights,
    }

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)

    elapsed = time.time() - start_ts
    print(
        f"全部完成（耗時 {elapsed:.1f} 秒）｜股票池 {len(tickers)} 隻｜"
        f"突破 {len(candidates['breakouts'])} 隻｜反轉 {len(candidates['reversals'])} 隻｜"
        f"提示 {len(insights['alerts'])} 則"
    )


if __name__ == "__main__":
    main()
