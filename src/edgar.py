"""
Pull 10-K filings from SEC EDGAR and extract the Risk Factors section.

Flow:
  ticker -> CIK            (company_tickers.json)
  CIK    -> filing index   (data.sec.gov/submissions, with pagination)
  filing -> primary doc    (www.sec.gov/Archives/...)
  doc    -> Item 1A text   (HTML -> text -> section slice)

Everything is cached under data/raw/ so a re-run is offline and the filing
vintage is frozen. Set SEC_USER_AGENT to your own "name email".
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import (
    DEFAULT_USER_AGENT,
    EDGAR_REQUEST_PAUSE,
    FIRST_FISCAL_YEAR,
    FORM_TYPE,
    LAST_FISCAL_YEAR,
    MIN_SECTION_CHARS,
    TICKERS,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
FILINGS = RAW / "filings"
FILINGS.mkdir(parents=True, exist_ok=True)

UA = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})


def _get(url: str, **kw) -> requests.Response:
    r = _SESSION.get(url, timeout=30, **kw)
    r.raise_for_status()
    time.sleep(EDGAR_REQUEST_PAUSE)
    return r


# --------------------------------------------------------------------------- #
# ticker -> CIK
# --------------------------------------------------------------------------- #
# SEC's company_tickers.json occasionally maps a ticker to a newer shell entity
# rather than the operating company we mean. Pin the ones in our universe that
# are ambiguous.
_CIK_OVERRIDE = {
    "XOM": "0000034088",   # Exxon Mobil Corp
    "GOOGL": "0001652044",  # Alphabet Inc.
}


def ticker_cik_map() -> dict[str, str]:
    cache = RAW / "company_tickers.json"
    if not cache.exists():
        r = _get("https://www.sec.gov/files/company_tickers.json")
        cache.write_text(r.text)
    raw = json.loads(cache.read_text())
    out = {}
    for row in raw.values():
        out[row["ticker"].upper()] = f"{int(row['cik_str']):010d}"
    out.update(_CIK_OVERRIDE)
    return out


# --------------------------------------------------------------------------- #
# CIK -> list of 10-K filings
# --------------------------------------------------------------------------- #
def _submissions(cik10: str) -> dict:
    cache = RAW / f"submissions_{cik10}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    data = _get(f"https://data.sec.gov/submissions/CIK{cik10}.json").json()
    # older filings live in paginated files referenced under filings.files
    extra_frames = []
    for f in data.get("filings", {}).get("files", []):
        ext = _get(f"https://data.sec.gov/submissions/{f['name']}").json()
        extra_frames.append(ext)
    data["_extra"] = extra_frames
    cache.write_text(json.dumps(data))
    return data


def list_10k_filings(ticker: str, cik10: str) -> pd.DataFrame:
    data = _submissions(cik10)

    def _rows(block: dict) -> pd.DataFrame:
        if not block:
            return pd.DataFrame()
        return pd.DataFrame({k: block[k] for k in
                             ["form", "filingDate", "reportDate",
                              "accessionNumber", "primaryDocument"]})

    frames = [_rows(data.get("filings", {}).get("recent", {}))]
    frames += [_rows(b) for b in data.get("_extra", [])]
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    df = df[df["form"].isin([FORM_TYPE, f"{FORM_TYPE}/A"])].copy()
    df["filingDate"] = pd.to_datetime(df["filingDate"])
    df["reportDate"] = pd.to_datetime(df["reportDate"], errors="coerce")
    df["fiscal_year"] = df["reportDate"].dt.year
    df = df[
        (df["fiscal_year"] >= FIRST_FISCAL_YEAR)
        & (df["fiscal_year"] <= LAST_FISCAL_YEAR)
        & (df["form"] == FORM_TYPE)          # drop amendments for the primary panel
    ]
    # one 10-K per fiscal year (first filed)
    df = df.sort_values("filingDate").drop_duplicates("fiscal_year", keep="first")
    df["ticker"] = ticker
    df["cik"] = cik10
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# filing -> primary document HTML
# --------------------------------------------------------------------------- #
def _doc_url(cik10: str, accession: str, primary_doc: str) -> str:
    acc_nodash = accession.replace("-", "")
    cik_int = int(cik10)
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_doc}"


def fetch_filing_html(row: pd.Series) -> str:
    dest = FILINGS / f"{row['ticker']}_{row['fiscal_year']}.html"
    if dest.exists():
        return dest.read_text(errors="ignore")
    url = _doc_url(row["cik"], row["accessionNumber"], row["primaryDocument"])
    html = _get(url).text
    dest.write_text(html)
    return html


# --------------------------------------------------------------------------- #
# HTML -> Item 1A text
# --------------------------------------------------------------------------- #
# Match section *headers*: "Item 1A" immediately followed by the section's
# canonical title. This ignores the many in-text cross-references
# ("as described in Part I, Item 1A ...") that a bare "Item 1A" match picks up.
_SEP = r"[\.\:\)\s\u2014\-]{0,4}"
_HEADER_RE = {
    "item_1a": re.compile(_SEP.join([r"item", r"1a", r"risk factors"]), re.I),
    "item_1b": re.compile(_SEP.join([r"item", r"1b", r"unresolved staff comments"]), re.I),
    "item_2": re.compile(_SEP.join([r"item", r"2", r"properties"]), re.I),
    "item_7": re.compile(_SEP.join([r"item", r"7", r"management.s discussion"]), re.I),
    "item_7a": re.compile(_SEP.join([r"item", r"7a", r"quantitative and qualitative"]), re.I),
    "item_8": re.compile(_SEP.join([r"item", r"8", r"financial statements"]), re.I),
}


def html_to_text(html: str) -> str:
    """HTML -> newline-delimited text. Uses lxml directly (BeautifulSoup's
    get_text is ~10x slower on the 1MB+ bank filings). Long tables (financial
    statements, exhibit lists) are dropped; short ones are kept because older
    filings lay section headers out as one-row tables."""
    import lxml.html

    try:
        root = lxml.html.fromstring(html)
        for el in root.iter("script", "style"):
            if el.getparent() is not None:
                el.getparent().remove(el)
        for tbl in root.iter("table"):
            if len(tbl.text_content()) > 400 and tbl.getparent() is not None:
                tbl.getparent().remove(tbl)
        # insert newlines at block boundaries so headers land on their own line
        for el in root.iter("p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4"):
            el.tail = (el.tail or "") + "\n"
        text = root.text_content()
    except Exception:
        text = BeautifulSoup(html, "lxml").get_text("\n")

    # normalise whitespace -- runs on every path
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\u200e", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]*\n+", "\n", text)
    return text


# A heading is NOT preceded by see/in/within/and-the... and is NOT immediately
# followed by a dash, comma, quote, lowercase word, or "section" -- those mark a
# cross-reference like "see the Item 1A. Risk Factors-Global Operations section".
_PRECEDE_BAD = re.compile(r"\b(see|refer|in|within|under|and|to|of|the)\s+(the\s+)?$", re.I)
_FOLLOW_BAD = re.compile(r"^[\s]*([—–\-,;:\"']|section\b|sections\b|and\b|or\b|above\b|below\b|in\s|on\s|[a-z])")


def _real_headers(text: str, key: str) -> list[int]:
    out = []
    for m in _HEADER_RE[key].finditer(text):
        before = text[max(0, m.start() - 16):m.start()].replace("\n", " ")
        after = text[m.end():m.end() + 40].replace("\n", " ", 1)
        if _PRECEDE_BAD.search(before):
            continue
        if _FOLLOW_BAD.match(after):
            continue
        out.append(m.start())
    return out


def extract_section(
    text: str,
    start_key: str,
    end_keys: tuple,
    min_chars: int = 1_500,
    max_frac: float = 0.85,
) -> str:
    """Return the body of a 10-K item.

    A 10-K names each item many times: in the table of contents, once as the
    real section heading, and repeatedly as cross-references. `_real_headers`
    filters to genuine headings; among the survivors we take each start heading,
    slice to the first end heading after it, and keep the largest span that is
    still a plausible section (>= `min_chars`, <= `max_frac` of the document).
    """
    starts = _real_headers(text, start_key)
    ends = sorted(e for k in end_keys for e in _real_headers(text, k))
    if not starts or not ends:
        return ""
    cap = int(max_frac * len(text))
    best = ""
    for s in starts:
        nxt = [e for e in ends if e > s]
        if not nxt:
            continue
        span = text[s:nxt[0]].strip()
        if min_chars <= len(span) <= cap and len(span) > len(best):
            best = span
    return best
