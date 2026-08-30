"""
Configuration: the firm universe, the sample window, section targets, and the
knobs for the sentiment models and the forward-return windows.

Design in one paragraph: for a fixed panel of large-cap US firms, pull every
annual report (10-K) 2015-2024, extract the "Risk Factors" section (Item 1A),
score its tone two ways -- FinBERT (a finance-domain transformer) and the
Loughran-McDonald finance sentiment dictionary (a transparent word-count
baseline) -- and test whether that tone, and especially the year-over-year
*change* in tone, predicts forward stock returns and forward realized
volatility. FinBERT is only interesting here if it beats the dictionary.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sample window
# ---------------------------------------------------------------------------
# Fiscal years whose 10-K we analyze. A 2014 FY 10-K is typically filed in
# early 2015; we key everything off the actual filing date from EDGAR.
FIRST_FISCAL_YEAR = 2014
LAST_FISCAL_YEAR = 2023

PRICE_START = "2013-06-01"   # enough lead-in to compute pre-filing volatility
PRICE_END = "2025-06-01"     # enough follow for a 12-month forward window on the last filing

MARKET_TICKER = "SPY"        # market proxy for abnormal returns

# ---------------------------------------------------------------------------
# Firm universe
# ---------------------------------------------------------------------------
# ~45 large caps chosen for sector spread and a long continuous 10-K history
# over the window. CIKs are resolved at runtime from SEC's company_tickers.json
# (see edgar.py) so this stays a plain ticker list.
UNIVERSE = {
    # Technology / communication
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "ORCL": "Technology", "CSCO": "Technology", "ADBE": "Technology",
    "CRM": "Technology", "INTC": "Technology", "IBM": "Technology",
    "GOOGL": "Communication", "META": "Communication", "NFLX": "Communication",
    "DIS": "Communication", "T": "Communication", "VZ": "Communication",
    # Consumer
    "AMZN": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    # Health care
    "JNJ": "Health Care", "PFE": "Health Care", "MRK": "Health Care",
    "ABBV": "Health Care", "UNH": "Health Care", "TMO": "Health Care",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "AXP": "Financials",
    # Industrials / energy / materials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "HON": "Industrials", "UPS": "Industrials",
    "XOM": "Energy", "CVX": "Energy",
}

TICKERS = sorted(UNIVERSE)
SECTORS = dict(UNIVERSE)

# ---------------------------------------------------------------------------
# Filing / section extraction
# ---------------------------------------------------------------------------
FORM_TYPE = "10-K"
SECTION = "item_1a"           # Risk Factors
# fallbacks if 1A can't be delimited cleanly
SECTION_FALLBACK = "item_7"   # MD&A

# A 10-K Item 1A under ~1,500 chars is almost always a cross-reference stub
# ("see Item 1A of our 2019 Form 10-K"), not the real section -> drop it.
MIN_SECTION_CHARS = 1_500

# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"          # labels: positive / negative / neutral
FINBERT_MAX_TOKENS = 256                     # risk-factor sentences rarely exceed this
FINBERT_BATCH_SIZE = 16
# Cap sentences scored per filing. Risk sections run 200-800 sentences; the tone
# measure is a mean over sentences and is stable well before all of them, so we
# score an even-strided sample. None = all (slow on CPU).
MAX_SENTENCES_PER_FILING = 120

# ---------------------------------------------------------------------------
# Forward-return / volatility windows (trading days after the filing date)
# ---------------------------------------------------------------------------
# Skip the first day after filing to avoid the announcement micro-window.
EVENT_GAP_DAYS = 1
FORWARD_WINDOWS = {
    "ret_1m": 21,
    "ret_3m": 63,
    "ret_6m": 126,
}
VOL_WINDOW_DAYS = 63          # forward realized-vol horizon
PRE_VOL_WINDOW_DAYS = 63      # trailing realized-vol baseline (ends at filing date)
PRE_RET_WINDOW_DAYS = 126     # trailing return control (momentum / reversal)

# ---------------------------------------------------------------------------
# SEC EDGAR access
# ---------------------------------------------------------------------------
# SEC requires a descriptive User-Agent with contact info on every request.
# Override with the SEC_USER_AGENT environment variable (recommended: your own
# "name email"). This default is deliberately generic and not tied to a person.
DEFAULT_USER_AGENT = "finbert-10k-sentiment research project contact@example.com"
EDGAR_REQUEST_PAUSE = 0.20    # seconds between EDGAR requests (SEC asks < 10/s)
