"""
Turn a daily price panel + a filing date into forward and trailing
return/volatility features for that filing.

All windows are in trading days relative to the filing date. The first
EVENT_GAP_DAYS after filing are skipped so the measurement window starts after
the initial filing-day reaction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    EVENT_GAP_DAYS,
    FORWARD_WINDOWS,
    MARKET_TICKER,
    PRE_RET_WINDOW_DAYS,
    PRE_VOL_WINDOW_DAYS,
    VOL_WINDOW_DAYS,
)

TRADING_DAYS_YR = 252


def _series(panel: pd.DataFrame, ticker: str) -> pd.DataFrame:
    s = panel[panel["ticker"] == ticker].sort_values("date").reset_index(drop=True)
    return s


def _pos_on_or_after(dates: pd.Series, d: pd.Timestamp) -> int | None:
    idx = dates.searchsorted(d, side="left")
    return int(idx) if idx < len(dates) else None


def filing_features(price_panel: pd.DataFrame, ticker: str, filing_date: pd.Timestamp) -> dict:
    """Forward/trailing features for one (ticker, filing_date).

    Forward returns are cumulative simple returns; the *abnormal* version
    subtracts the market (SPY) cumulative return over the identical calendar
    window. Volatility is annualised daily-return standard deviation.
    """
    stock = _series(price_panel, ticker)
    mkt = _series(price_panel, MARKET_TICKER)
    if stock.empty:
        return {}

    d0 = _pos_on_or_after(stock["date"], filing_date)
    if d0 is None:
        return {}
    start = d0 + EVENT_GAP_DAYS
    if start >= len(stock):
        return {}

    out: dict = {"price_date": stock["date"].iloc[d0]}
    sret = stock["ret"].to_numpy()
    mdate_to_ret = dict(zip(mkt["date"], mkt["ret"]))

    def cum(arr: np.ndarray) -> float:
        arr = arr[~np.isnan(arr)]
        return float(np.prod(1.0 + arr) - 1.0) if len(arr) else np.nan

    # forward simple + abnormal returns
    for name, horizon in FORWARD_WINDOWS.items():
        end = start + horizon
        if end > len(stock):
            out[name] = np.nan
            out[f"a{name}"] = np.nan
            continue
        window_dates = stock["date"].iloc[start:end]
        r_stock = cum(sret[start:end])
        r_mkt = cum(np.array([mdate_to_ret.get(dt, np.nan) for dt in window_dates]))
        out[name] = r_stock
        out[f"a{name}"] = r_stock - r_mkt  # abnormal = stock - market

    # forward realised volatility
    end_v = start + VOL_WINDOW_DAYS
    fwd_v = sret[start:end_v]
    fwd_v = fwd_v[~np.isnan(fwd_v)]
    out["fwd_vol"] = float(np.std(fwd_v, ddof=1) * np.sqrt(TRADING_DAYS_YR)) if len(fwd_v) > 5 else np.nan

    # trailing controls (end at the filing day, exclusive of the gap)
    pv0 = max(d0 - PRE_VOL_WINDOW_DAYS, 0)
    pre_v = sret[pv0:d0]
    pre_v = pre_v[~np.isnan(pre_v)]
    out["pre_vol"] = float(np.std(pre_v, ddof=1) * np.sqrt(TRADING_DAYS_YR)) if len(pre_v) > 5 else np.nan
    out["vol_change"] = out["fwd_vol"] - out["pre_vol"] if pd.notna(out["fwd_vol"]) and pd.notna(out["pre_vol"]) else np.nan

    pr0 = max(d0 - PRE_RET_WINDOW_DAYS, 0)
    out["pre_ret_6m"] = cum(sret[pr0:d0])

    # liquidity / size proxy: log median daily dollar volume over the trailing year
    tail = stock.iloc[max(d0 - TRADING_DAYS_YR, 0):d0]
    dv = (tail["close"] * tail["volume"]).dropna()
    out["ln_dollar_vol"] = float(np.log(dv.median())) if len(dv) else np.nan

    return out
