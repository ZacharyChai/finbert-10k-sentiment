"""
Daily adjusted-close prices from the public Yahoo Finance chart endpoint.

No API key. One CSV per ticker cached under data/raw/prices/. We only need
adjusted close (splits + dividends folded in) to compute total returns.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from config import MARKET_TICKER, PRICE_END, PRICE_START, TICKERS

PRICES = Path(__file__).resolve().parents[1] / "data" / "raw" / "prices"
PRICES.mkdir(parents=True, exist_ok=True)

_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
_HEADERS = {"User-Agent": "Mozilla/5.0 (research; price history)"}


def _fetch_one(ticker: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp(PRICE_START).timestamp())
    p2 = int(pd.Timestamp(PRICE_END).timestamp())
    r = requests.get(
        _CHART.format(ticker=ticker),
        params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,split"},
        headers=_HEADERS, timeout=30,
    )
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ts = pd.to_datetime(res["timestamp"], unit="s").tz_localize(None).normalize()
    ind = res["indicators"]
    adj = ind.get("adjclose", [{}])[0].get("adjclose")
    quote = ind["quote"][0]
    close = quote["close"]
    df = pd.DataFrame(
        {
            "date": ts,
            "adj_close": adj if adj is not None else close,
            "close": close,
            "volume": quote.get("volume"),
        }
    )
    df = df[~df["date"].duplicated(keep="last")].dropna(subset=["adj_close"])
    return df.reset_index(drop=True)


def get_prices(ticker: str, force: bool = False) -> pd.DataFrame:
    cache = PRICES / f"{ticker}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, parse_dates=["date"])
    df = _fetch_one(ticker)
    df.to_csv(cache, index=False)
    time.sleep(0.4)
    return df


def build_price_panel(force: bool = False) -> pd.DataFrame:
    frames = []
    for t in TICKERS + [MARKET_TICKER]:
        df = get_prices(t, force=force)
        df["ticker"] = t
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    panel["ret"] = panel.groupby("ticker")["adj_close"].pct_change()
    panel["logret"] = np.log1p(panel["ret"])
    return panel.reset_index(drop=True)


if __name__ == "__main__":
    import sys

    panel = build_price_panel(force="--force" in sys.argv)
    out = Path(__file__).resolve().parents[1] / "data" / "processed" / "prices.csv"
    panel.to_csv(out, index=False)
    print(f"{panel['ticker'].nunique()} tickers, "
          f"{panel['date'].min():%Y-%m-%d}..{panel['date'].max():%Y-%m-%d}, "
          f"{len(panel):,} rows -> {out}")
