PY ?= .venv/bin/python
DOCKER ?= sudo docker
COMPOSE ?= $(DOCKER) compose
MLFLOW_TRACKING_URI ?= http://127.0.0.1:5055
export MLFLOW_TRACKING_URI

.PHONY: download-data train replay load-test inject-drift test lint up down benchmark mlflow

download-data:
	$(PY) -m surgecast.data.download

mlflow:
	mkdir -p mlartifacts data/raw data/processed data/runtime
	touch mlflow.db
	$(PY) -m mlflow server --host 0.0.0.0 --port 5055 \
		--backend-store-uri sqlite:///$(CURDIR)/mlflow.db \
		--default-artifact-root $(CURDIR)/mlartifacts

train:
	$(PY) -m surgecast.training.train

replay:
	$(PY) -m surgecast.streaming.replay --speed $${SPEED:-0} --max-events $${MAX_EVENTS:-0}

inject-drift:
	$(PY) -m surgecast.streaming.shock --multiplier $${SHOCK:-3} \
		--start $${START:-2024-01-29T00:00:00} --end $${END:-2024-01-29T12:00:00}

load-test:
	mkdir -p results/locust
	.venv/bin/locust -f locustfile.py --headless -u $${USERS:-30} -r $${SPAWN:-10} \
		-t $${DURATION:-30s} --host $${HOST:-http://127.0.0.1:8841} --csv results/locust/run

test:
	$(PY) -m pytest

lint:
	.venv/bin/ruff check surgecast tests

up:
	mkdir -p mlartifacts data/raw data/processed data/runtime
	touch mlflow.db
	HOST_ARTIFACT_DIR=$(CURDIR)/mlartifacts $(COMPOSE) up -d --build

down:
	$(COMPOSE) down

benchmark:
	$(PY) -m surgecast.bench.run
