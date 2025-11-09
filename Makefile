PY=python

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

format:
	black src && isort src

lint:
	flake8 src

run-fl:
	bash scripts/run_fl_sim.sh

download-data:
	bash scripts/download_dataset.sh