"""
Two tone scorers for a block of filing text:

  FinBERTScorer  -- ProsusAI/finbert, a BERT fine-tuned on financial phrasebank.
                    Per-sentence P(positive/negative/neutral), aggregated.
  lm_score       -- Loughran-McDonald finance sentiment dictionary (via
                    pysentiment2). Transparent word-count baseline.

FinBERT is only worth the dependency if it carries signal the dictionary misses,
so both are computed for every filing and compared in the analysis.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from config import FINBERT_BATCH_SIZE, FINBERT_MAX_TOKENS, FINBERT_MODEL

# --------------------------------------------------------------------------- #
# sentence splitting
# --------------------------------------------------------------------------- #
_ABBREV = r"(?<!\b[A-Z]\.)(?<!\bInc\.)(?<!\bCorp\.)(?<!\bCo\.)(?<!\bLtd\.)(?<!\bU\.S\.)(?<!\bvs\.)(?<!\bNo\.)"
_SENT_END = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z(\"'])")


def split_sentences(text: str, min_chars: int = 25, max_chars: int = 700) -> list[str]:
    parts: list[str] = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        for s in _SENT_END.split(para):
            s = re.sub(r"\s+", " ", s).strip()
            if not (min_chars <= len(s) <= max_chars):
                continue
            # drop junk that leaked from tables / TOCs: needs to be mostly letters
            alpha = sum(c.isalpha() or c.isspace() for c in s)
            if alpha / len(s) < 0.75:
                continue
            parts.append(s)
    return parts


# --------------------------------------------------------------------------- #
# Loughran-McDonald dictionary baseline
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _lm():
    import pysentiment2 as ps

    return ps.LM()


@dataclass
class LMScore:
    n_tokens: int
    n_negative: int
    n_positive: int
    neg_frac: float          # negative words / total tokens
    pos_frac: float
    net_tone: float          # (pos - neg) / (pos + neg), in [-1, 1]; 0 if none
    polarity: float          # pysentiment2's own polarity measure

    def as_dict(self) -> dict:
        return {f"lm_{k}": v for k, v in self.__dict__.items()}


def lm_score(text: str) -> LMScore:
    lm = _lm()
    tokens = lm.tokenize(text)
    sc = lm.get_score(tokens)
    n = max(len(tokens), 1)
    neg, pos = int(sc["Negative"]), int(sc["Positive"])
    net = (pos - neg) / (pos + neg) if (pos + neg) else 0.0
    return LMScore(
        n_tokens=len(tokens),
        n_negative=neg,
        n_positive=pos,
        neg_frac=neg / n,
        pos_frac=pos / n,
        net_tone=net,
        polarity=float(sc["Polarity"]),
    )


# --------------------------------------------------------------------------- #
# FinBERT
# --------------------------------------------------------------------------- #
@dataclass
class FinBERTScore:
    n_sentences: int
    p_positive: float        # mean P(positive) across sentences
    p_negative: float
    p_neutral: float
    net_sentiment: float     # mean(P(pos) - P(neg))   -- the headline tone measure
    pct_negative: float      # share of sentences whose argmax label is negative
    pct_positive: float
    pct_neutral: float

    def as_dict(self) -> dict:
        return {f"fb_{k}": v for k, v in self.__dict__.items()}


class FinBERTScorer:
    """Lazy-loads the model on first use so importing this module stays cheap."""

    _LABELS = ("positive", "negative", "neutral")

    def __init__(self, model_name: str = FINBERT_MODEL, device: str | None = None):
        self.model_name = model_name
        self._device = device
        self._tok = None
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # physical-core count is the sweet spot; all logical cores thrashes
        torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 2 or 1)))
        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device).eval()
        # ProsusAI/finbert id2label: 0=positive, 1=negative, 2=neutral
        self._id2label = [self._model.config.id2label[i].lower()
                          for i in range(self._model.config.num_labels)]

    def score_sentences(self, sentences: list[str]) -> np.ndarray:
        """Return an (n, 3) array of probabilities in (positive, negative, neutral)
        column order, regardless of the model's internal label order.

        Sentences are sorted by length before batching so each batch pads to a
        similar width -- a few long sentences in an otherwise-short batch
        otherwise dominate CPU time.
        """
        self._ensure()
        torch = self._torch
        if not sentences:
            return np.empty((0, 3))
        col = [self._id2label.index(lbl) for lbl in self._LABELS]
        order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
        probs_sorted = np.empty((len(sentences), 3))
        for b in range(0, len(order), FINBERT_BATCH_SIZE):
            idx = order[b:b + FINBERT_BATCH_SIZE]
            enc = self._tok([sentences[i] for i in idx], return_tensors="pt",
                            padding=True, truncation=True, max_length=FINBERT_MAX_TOKENS)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                logits = self._model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()[:, col]
            for k, i in enumerate(idx):
                probs_sorted[i] = p[k]
        return probs_sorted

    @staticmethod
    def sample_sentences(text: str, max_sentences: int | None) -> list[str]:
        sents = split_sentences(text)
        if max_sentences and len(sents) > max_sentences:
            idx = np.linspace(0, len(sents) - 1, max_sentences).round().astype(int)
            sents = [sents[i] for i in sorted(set(idx))]
        return sents

    @staticmethod
    def _aggregate(probs: np.ndarray, n_sents: int) -> FinBERTScore:
        if len(probs) == 0:
            return FinBERTScore(0, *[float("nan")] * 7)
        pos, neg, neu = probs[:, 0], probs[:, 1], probs[:, 2]
        argmax = probs.argmax(axis=1)
        return FinBERTScore(
            n_sentences=n_sents,
            p_positive=float(pos.mean()),
            p_negative=float(neg.mean()),
            p_neutral=float(neu.mean()),
            net_sentiment=float((pos - neg).mean()),
            pct_negative=float((argmax == 1).mean()),
            pct_positive=float((argmax == 0).mean()),
            pct_neutral=float((argmax == 2).mean()),
        )

    def score_list(self, sentences: list[str]) -> FinBERTScore:
        """Score an already-split list of sentences (no re-splitting)."""
        return self._aggregate(self.score_sentences(sentences), len(sentences))

    def score_text(self, text: str, max_sentences: int | None = None) -> tuple[FinBERTScore, list[str]]:
        sents = self.sample_sentences(text, max_sentences)
        return self._aggregate(self.score_sentences(sents), len(sents)), sents
