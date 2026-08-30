"""Put src/ on sys.path so `import edgar`, `import sentiment`, ... work in tests
the same bare way the scripts import them when run directly."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
