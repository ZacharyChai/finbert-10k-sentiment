.PHONY: all data analysis test test-slow clean

PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python)

all: analysis test

data:
	$(PY) src/build_dataset.py

analysis:
	$(PY) src/analysis.py

test:
	$(PY) -m pytest

test-slow:
	$(PY) -m pytest -m slow

clean:
	rm -rf output/*.csv figures/*.png data/processed/analysis_frame.csv
	find . -name __pycache__ -type d -exec rm -rf {} +
