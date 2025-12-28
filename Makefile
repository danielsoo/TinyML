PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

format:
	black src && isort src

lint:
	flake8 src

run-fl:
	bash scripts/run_fl_sim.sh

train:
	python scripts/train.py

download-data:
	bash scripts/download_dataset.sh

analyze-compression:
	python scripts/analyze_compression.py --models $(MODELS) --config config/federated_local.yaml

visualize-results:
	python scripts/visualize_results.py --results data/processed/analysis/compression_analysis.csv