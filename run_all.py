"""
End-to-end pipeline.

    python run_all.py               # rebuild filings.csv from the committed
                                    # sentiment cache + prices, then analyse
    python run_all.py --refetch     # re-pull 10-Ks from EDGAR and re-score with
                                    # FinBERT first (slow: ~1-2h on CPU, ~440MB model)

Steps:
  1. build_dataset.py  -> data/processed/{prices.csv, filings.csv}
                          (uses data/processed/sentiment_cache/ if present)
  2. analysis.py       -> output/*.csv, figures/*.png
"""

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
PY = sys.executable


def run(script: str, *args: str) -> None:
    print(f"\n{'=' * 72}\n  {script} {' '.join(args)}\n{'=' * 72}", flush=True)
    subprocess.run([PY, str(SRC / script), *args], check=True, cwd=SRC)


if __name__ == "__main__":
    refetch = "--refetch" in sys.argv
    if refetch:
        # clearing the sentiment cache forces a fresh FinBERT pass
        for f in (SRC.parent / "data" / "processed" / "sentiment_cache").glob("*.json"):
            f.unlink()
    run("build_dataset.py")
    run("analysis.py")
    print("\nDone. See MEMO.md for the written interpretation.")
