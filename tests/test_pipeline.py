"""Unit tests for the pieces that have logic worth pinning:
section extraction, sentence splitting, the LM scorer, the return/vol features,
and the year-over-year change rule. FinBERT itself is exercised in one slow,
opt-in test.

Ordering follows the pipeline: edgar -> sentiment -> returns -> build_dataset.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import edgar
import sentiment
from returns import filing_features

# --------------------------------------------------------------------------- #
# edgar.extract_section / _real_headers
# --------------------------------------------------------------------------- #
_MINI_10K = """
PART I
Item 1. Business
We make widgets and sell them worldwide.
Item 1A. Risk Factors
The following discussion sets forth the material risks to our business.
A downturn in demand could materially and adversely affect our revenue and results.
We depend on a small number of suppliers and the loss of one would harm operations.
Item 1B. Unresolved Staff Comments
None.
Item 2. Properties
We lease our headquarters.
Item 7. Management's Discussion and Analysis
As described in Part I, Item 1A. Risk Factors, macro conditions remain uncertain.
"""


def test_extract_section_grabs_the_real_item_1a_not_the_toc_or_cross_reference():
    text = edgar.html_to_text(f"<html><body><pre>{_MINI_10K}</pre></body></html>")
    sec = edgar.extract_section(text, "item_1a", ("item_1b", "item_2"), min_chars=50)
    assert sec.lower().startswith("item 1a")
    assert "material risks to our business" in sec
    assert "small number of suppliers" in sec
    # must stop at Item 1B, not run into MD&A
    assert "Management's Discussion" not in sec
    assert "macro conditions remain uncertain" not in sec


def test_real_headers_rejects_an_inline_cross_reference():
    text = ("... for more detail see the Item 1A. Risk Factors section of this report. "
            "\nItem 1A. Risk Factors\nThe real section body starts here and is long enough.")
    hits = edgar._real_headers(text, "item_1a")
    # only the true heading (preceded by newline, followed by a capitalised line)
    assert len(hits) == 1
    assert text[hits[0]:].startswith("Item 1A. Risk Factors\nThe real section")


def test_extract_section_returns_empty_when_no_end_header_present():
    text = "Item 1A. Risk Factors\nSome risks here but the document is truncated."
    assert edgar.extract_section(text, "item_1a", ("item_1b", "item_2")) == ""


# --------------------------------------------------------------------------- #
# sentiment.split_sentences
# --------------------------------------------------------------------------- #
def test_split_sentences_breaks_on_terminal_punctuation_not_abbreviations():
    text = ("Our competitor, Acme Corp., filed suit in the U.S. District Court. "
            "We believe the claims lack merit. A loss could still be material.")
    sents = sentiment.split_sentences(text, min_chars=5)
    assert len(sents) == 3
    assert sents[0].endswith("District Court.")
    assert sents[1] == "We believe the claims lack merit."


def test_split_sentences_filters_by_length():
    text = "Too short. " + "This sentence is comfortably within the length window and is kept."
    sents = sentiment.split_sentences(text, min_chars=25)
    assert all(len(s) >= 25 for s in sents)
    assert any("length window" in s for s in sents)


# --------------------------------------------------------------------------- #
# sentiment.lm_score  (Loughran-McDonald dictionary)
# --------------------------------------------------------------------------- #
def test_lm_score_is_negative_for_risk_heavy_text_and_positive_for_upbeat_text():
    neg = ("The adverse litigation and impairment losses could materially harm our "
           "results, and a default or downturn would damage our reputation.")
    pos = ("Strong demand and successful new products drove exceptional growth and "
           "improved profitability, and we achieved record gains.")
    s_neg = sentiment.lm_score(neg)
    s_pos = sentiment.lm_score(pos)
    assert s_neg.net_tone < 0 < s_pos.net_tone
    assert s_neg.n_negative >= 3
    assert s_pos.n_positive >= 3


def test_lm_score_neutral_text_has_zero_net_tone():
    s = sentiment.lm_score("The company operates in three geographic segments and reports in US dollars.")
    assert s.net_tone == 0.0
    assert s.n_negative == 0 and s.n_positive == 0


# --------------------------------------------------------------------------- #
# returns.filing_features
# --------------------------------------------------------------------------- #
def _price_panel(ticker_ret: float, mkt_ret: float, n: int = 400) -> pd.DataFrame:
    """A panel where `ticker` compounds at ticker_ret/day and SPY at mkt_ret/day."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    frames = []
    for tk, r in [("TEST", ticker_ret), ("SPY", mkt_ret)]:
        px = 100 * np.cumprod(np.r_[1.0, np.full(n - 1, 1 + r)])
        d = pd.DataFrame({"date": dates, "adj_close": px, "close": px,
                          "volume": 1_000_000, "ticker": tk})
        d["ret"] = d["adj_close"].pct_change()
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def test_filing_features_abnormal_return_nets_out_the_market():
    panel = _price_panel(ticker_ret=0.001, mkt_ret=0.001)  # move identically
    f = filing_features(panel, "TEST", pd.Timestamp("2020-03-01"))
    # raw forward return is positive, abnormal (stock - market) is ~0
    assert f["ret_1m"] > 0
    assert abs(f["aret_1m"]) < 1e-6


def test_filing_features_forward_vol_is_zero_for_a_constant_growth_series():
    panel = _price_panel(ticker_ret=0.0005, mkt_ret=0.0)
    f = filing_features(panel, "TEST", pd.Timestamp("2020-02-01"))
    assert f["fwd_vol"] == pytest.approx(0.0, abs=1e-9)
    # market return is zero here, so abnormal return equals the raw return
    assert f["aret_1m"] == pytest.approx(f["ret_1m"], abs=1e-9)
    assert f["ret_1m"] > 0


def test_filing_features_returns_empty_when_filing_date_past_price_history():
    panel = _price_panel(0.0, 0.0, n=100)
    assert filing_features(panel, "TEST", pd.Timestamp("2025-01-01")) == {}


# --------------------------------------------------------------------------- #
# build_dataset.add_yoy_changes
# --------------------------------------------------------------------------- #
def test_add_yoy_changes_only_differences_consecutive_fiscal_years():
    import build_dataset as bd

    df = pd.DataFrame({
        "ticker": ["A", "A", "A", "B", "B"],
        "fiscal_year": [2015, 2016, 2018, 2016, 2017],   # A skips 2017
        "fb_net_sentiment": [0.0, -0.2, -0.5, 0.1, 0.0],
        "fb_pct_negative": [0.3, 0.4, 0.5, 0.2, 0.25],
        "fb_p_negative": [0.3, 0.4, 0.5, 0.2, 0.25],
        "lm_neg_frac": [0.01, 0.02, 0.03, 0.01, 0.012],
        "lm_net_tone": [0.0, -0.1, -0.2, 0.0, -0.05],
    })
    out = bd.add_yoy_changes(df)
    a = out[out.ticker == "A"].set_index("fiscal_year")
    assert pd.isna(a.loc[2015, "d_fb_net_sentiment"])          # no prior year
    assert a.loc[2016, "d_fb_net_sentiment"] == pytest.approx(-0.2)
    assert pd.isna(a.loc[2018, "d_fb_net_sentiment"])          # 2017 missing -> gap
    b = out[out.ticker == "B"].set_index("fiscal_year")
    assert b.loc[2017, "d_lm_neg_frac"] == pytest.approx(0.002)


# --------------------------------------------------------------------------- #
# FinBERT: slow, opt-in (needs the ~440MB model)
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_finbert_orders_sentiment_as_expected():
    scorer = sentiment.FinBERTScorer()
    probs = scorer.score_sentences([
        "The company reported record revenue and raised full-year guidance.",
        "The company disclosed a material weakness and expects significant losses.",
        "The company is headquartered in Delaware.",
    ])
    assert probs.shape == (3, 3)               # (pos, neg, neu) columns
    assert probs[0, 0] > probs[0, 1]           # sentence 1: positive > negative
    assert probs[1, 1] > probs[1, 0]           # sentence 2: negative > positive
    assert probs[2, 2] == pytest.approx(probs[2].max())  # sentence 3: neutral wins
