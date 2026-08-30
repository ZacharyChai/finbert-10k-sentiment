# Does the tone of a 10-K's risk section tell you anything about the stock?

**Short answer:** It lines up with the stock's *volatility*, not its *return* — and
even the volatility link is mostly the volatility the stock already had. A
finance-tuned language model (FinBERT) measures the tone just fine, but it does
not beat a 1990s-style word list at any of the prediction tasks here.

---

## What was built

For 41 large-cap US firms, every 10-K annual report for fiscal years
2014–2023 (**376 filings**) was pulled from SEC EDGAR. From each, the **Risk
Factors** section (Item 1A) was extracted and its tone scored two ways:

- **FinBERT** (`ProsusAI/finbert`) — a BERT model fine-tuned on financial text.
  Each sentence is classified positive / negative / neutral; the filing's tone
  is the average net sentiment across a 120-sentence sample.
- **Loughran–McDonald** — the standard finance sentiment dictionary. Count
  negative and positive words; the tone is the negative-word fraction. This is
  the transparent baseline FinBERT has to beat.

Each filing was then linked to the stock's forward **abnormal return** (return
minus the S&P 500) over 1, 3 and 6 months, and its forward **realised
volatility** over the next quarter, keyed to the actual filing date.

## Finding 1 — the tone measures are real, but they measure the present

Both measures pass a construct-validity check: a firm coming off a bad year
writes a more negative risk section. The tie to recent **volatility** is strong
and unambiguous — a one-standard-deviation higher trailing realised vol goes
with more negative tone on every measure (t ≈ −4.4 for FinBERT net sentiment,
+4.2 / +4.0 for FinBERT %-negative and LM negativity, all p < 0.001). The tie to
recent **returns** is weaker: a worse trailing 6-month return goes with more
negative FinBERT tone (t ≈ −2.2, p = 0.03) and, more marginally, higher LM
negativity (t ≈ 1.9, p = 0.05). So the text is not noise — but it is largely a
restatement of what already happened, which matters for what follows.

FinBERT and the dictionary agree only moderately: the filing-level correlation
between FinBERT's share-of-negative-sentences and the LM negative-word fraction
is **0.57**. They measure related but distinct things — about a third of the
variation is shared.

## Finding 2 — tone does not predict forward returns

Across 25 predictive regressions (five tone measures × five forward outcomes,
each with firm and year fixed effects and two-way clustered standard errors),
**zero** are significant at the 5% level. You would expect between one and two
significant by chance. The closest calls are a positive — wrong-signed —
6-month-return coefficient on FinBERT net sentiment (p = 0.10) and a
right-signed one on LM negativity (p = 0.07); neither is robust.

The within-year tercile portfolio sort tells the same story: sorting firms into
thirds by risk-factor tone and comparing the most-negative third to the
least-negative third, the forward-return spreads are all statistically
indistinguishable from zero (|t| < 1.9 at every horizon, for both measures).

A placebo check supports reading this as a real null rather than a
power problem in the wrong place: regressing the *trailing* 6-month return on
tone (i.e. asking whether tone "predicts" the past) also produces nothing
significant (all p > 0.24). Tone is correlated with recent volatility, but it is
not just a lagged-return proxy in disguise.

## Finding 3 — tone tracks forward volatility, but mostly mechanically

The one place tone lines up with the future is realised volatility. In the
tercile sort, firms with the most negative risk sections (LM negativity) have
next-quarter annualised volatility of **0.31** versus **0.24** for the least
negative — a monotonic rise across the three groups (0.24 → 0.27 → 0.31), a gap
of about 7 volatility points, **t ≈ 3.5**. FinBERT's net-sentiment sort points
the same way (more positive tone → lower forward vol) but only marginally
(t ≈ −1.9).

And this is largely the persistence of volatility, not new information in the
text. Negative tone reflects *current* high volatility (Finding 1), and
volatility is autocorrelated, so a negative-tone firm tends to stay volatile.
In the regression — which controls for the stock's trailing realised vol — the
FinBERT-vol relationship is gone (p ≈ 0.54) and the LM-vol one is only marginal
(p ≈ 0.08). The incremental predictive content of the text, on top of just
knowing how volatile the stock has been, is small.

## Finding 4 — FinBERT does not beat the dictionary

Put both measures in the same regression and neither has significant incremental
predictive power over the other, for any outcome. FinBERT is more expensive to
run (a 440 MB model, minutes of GPU-less compute per hundred filings), needs
careful sentence sampling, and classifies most risk-factor sentences as
"neutral". For this task — filing-level tone linked to forward market outcomes —
the word list does the same job.

## What this does and doesn't rule out

**Rules out** (within this sample): a simple, tradeable signal from the *level*
or *year-over-year change* of 10-K risk-factor tone to forward abnormal returns,
whether measured by FinBERT or by dictionary.

**Does not rule out:**

- **Bigger, more targeted samples.** 376 firm-years of large, heavily-covered
  stocks is where mispricing is least likely. The Loughran–McDonald return
  result in the literature comes from tens of thousands of filings including
  small, thinly-covered firms.
- **Sentence- or topic-level signal.** This study uses one number per filing.
  *New* risk factors, or changes in specific risk categories (litigation,
  liquidity, supply chain), may carry information that filing-level averaging
  washes out.
- **Short-window drift.** The forward windows here start the day after filing.
  A filing-day or filing-week reaction is not measured.
- **A better model.** FinBERT is fine-tuned on analyst notes, not 10-K legalese.
  A model tuned on filing text, or a modern long-context LLM, might extract more.

## The one assumption that matters

That the **filing date** is when the market first sees this text. It is not:
10-Ks are filed 2–8 weeks after fiscal year-end, after the earnings release, and
much of the risk-factor language is carried over verbatim year to year. If the
informative content of the risk section is already priced by the time the 10-K
is filed — which is likely for large caps — then a forward-return test keyed to
the filing date is looking after the fact, and a null is what you would expect
regardless of whether the text "matters".

## Bottom line

Risk-factor tone is a valid measure of how a firm's year went, and it co-moves
with volatility because volatile firms write nervous risk sections and stay
volatile. It is not, in this sample, a forward-looking signal for returns, and
the transformer earns its keep only if the task is harder or the text is
messier than this.
