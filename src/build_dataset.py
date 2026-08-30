"""
Assemble the filing-level dataset:

  for every (firm, fiscal year) 10-K in the universe
    -> fetch the filing, extract Item 1A (Risk Factors)
    -> score its tone with FinBERT and with the Loughran-McDonald dictionary
    -> attach forward returns / forward realised volatility from the price panel
    -> add year-over-year change in each tone measure

Output: data/processed/filings.csv  (one row per firm-year)

Per-filing sentiment is cached under data/processed/sentiment_cache/ so the
expensive FinBERT pass is only ever run once per filing.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pandas as pd

from config import (
    MAX_SENTENCES_PER_FILING,
    MIN_SECTION_CHARS,
    SECTION,
    SECTION_FALLBACK,
    TICKERS,
)
from edgar import (
    extract_section,
    fetch_filing_html,
    html_to_text,
    list_10k_filings,
    ticker_cik_map,
)
from returns import filing_features
from sentiment import FinBERTScorer, lm_score, split_sentences

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SENT_CACHE = PROC / "sentiment_cache"
SENT_CACHE.mkdir(parents=True, exist_ok=True)

_END_KEYS = {
    "item_1a": ("item_1b", "item_2"),
    "item_7": ("item_7a", "item_8"),
}


def _section_text(html: str) -> tuple[str, str]:
    text = html_to_text(html)
    sec = extract_section(text, SECTION, _END_KEYS[SECTION])
    used = SECTION
    if len(sec) < MIN_SECTION_CHARS:
        alt = extract_section(text, SECTION_FALLBACK, _END_KEYS[SECTION_FALLBACK])
        if len(alt) > len(sec):
            sec, used = alt, SECTION_FALLBACK
    return sec, used


def _score_filing(ticker: str, fy: int, section_text: str, scorer: FinBERTScorer) -> dict:
    cache = SENT_CACHE / f"{ticker}_{fy}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    # score FinBERT and LM on the SAME sampled sentences so the two tone
    # measures are comparable and the LM tokenizer isn't run on 40k words
    all_sents = split_sentences(section_text)
    sample = scorer.sample_sentences(section_text, MAX_SENTENCES_PER_FILING)
    fb = scorer.score_list(sample)
    lm = lm_score(" ".join(sample))
    rec = {**fb.as_dict(), **lm.as_dict(), "n_sentences_total": len(all_sents)}
    cache.write_text(json.dumps(rec))
    return rec


def build(limit: int | None = None) -> pd.DataFrame:
    cik = ticker_cik_map()
    scorer = FinBERTScorer()

    rows = []
    tickers = TICKERS[:limit] if limit else TICKERS
    for i, t in enumerate(tickers, 1):
        if t not in cik:
            print(f"  !! no CIK for {t}, skipping")
            continue
        try:
            filings = list_10k_filings(t, cik[t])
        except Exception as e:  # noqa: BLE001
            print(f"  !! {t}: filing index failed ({e})")
            continue
        print(f"[{i}/{len(tickers)}] {t}: {len(filings)} 10-K filings")
        for _, f in filings.iterrows():
            try:
                html = fetch_filing_html(f)
                sec_text, used = _section_text(html)
                if len(sec_text) < MIN_SECTION_CHARS:
                    print(f"    {f['fiscal_year']}: section too short ({len(sec_text)} chars), skip")
                    continue
                scores = _score_filing(t, int(f["fiscal_year"]), sec_text, scorer)
                rows.append({
                    "ticker": t,
                    "fiscal_year": int(f["fiscal_year"]),
                    "filing_date": f["filingDate"],
                    "report_date": f["reportDate"],
                    "section_used": used,
                    "section_chars": len(sec_text),
                    "accession": f["accessionNumber"],
                    **scores,
                })
            except Exception as e:  # noqa: BLE001
                print(f"    {f['fiscal_year']}: FAILED ({type(e).__name__}: {e})")

    df = pd.DataFrame(rows).sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    return df


def add_yoy_changes(df: pd.DataFrame) -> pd.DataFrame:
    tone_cols = ["fb_net_sentiment", "fb_pct_negative", "fb_p_negative",
                 "lm_neg_frac", "lm_net_tone"]
    df = df.sort_values(["ticker", "fiscal_year"]).copy()
    for c in tone_cols:
        df[f"d_{c}"] = df.groupby("ticker")[c].diff()
        # only a "true" YoY change if the prior filing is the immediately prior FY
        prev_fy = df.groupby("ticker")["fiscal_year"].shift(1)
        df.loc[df["fiscal_year"] - prev_fy != 1, f"d_{c}"] = pd.NA
    return df


def attach_returns(df: pd.DataFrame, price_panel: pd.DataFrame) -> pd.DataFrame:
    feats = []
    for _, r in df.iterrows():
        f = filing_features(price_panel, r["ticker"], pd.Timestamp(r["filing_date"]))
        feats.append(f)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(feats)], axis=1)


def main() -> None:
    limit = None
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])

    from prices import build_price_panel

    print("== prices ==")
    price_panel = build_price_panel()
    price_panel.to_csv(PROC / "prices.csv", index=False)

    print("== filings + sentiment ==")
    df = build(limit=limit)
    print(f"  {len(df)} scored filings across {df['ticker'].nunique()} firms")

    df = add_yoy_changes(df)
    df = attach_returns(df, price_panel)

    df.to_csv(PROC / "filings.csv", index=False)
    print(f"wrote {PROC / 'filings.csv'}  ({df.shape[0]} rows x {df.shape[1]} cols)")
    print(df.groupby("section_used")["ticker"].count().to_string())


if __name__ == "__main__":
    main()
