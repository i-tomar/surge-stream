# Status

Packaging stop. The streaming stack is not finished. Nothing below is a benchmark result. Forecast error, latency, and throughput are not restated here. The only measured training file is `results/train_metrics.json`, written by `python -m surgecast.training.train` on this machine. `RESULTS.md` was not generated. The README resume block is still the placeholder.

## Done

- Python package for shared 5-minute zone features, TLC parquet prep, LightGBM training with last-value and seasonal-naive baselines, and an optional PyTorch MLP comparison that is not the served model.
- FastAPI app, Redis feature processor, Kafka/Redpanda replay and shock producer, Evidently drift check, and a retrain function that promotes an MLflow `champion` alias.
- Pytest for features, metrics, drift detection, retrain promotion, and the API hot-swap. `ruff check surgecast tests` was clean before the last code commit.
- Docker Compose file for Redpanda, Redis, MLflow, API, processor, drift monitor, Prometheus, Grafana, and a built map. Grafana dashboard JSON is provisioned in `infra/grafana/`.
- Leaflet map in `frontend/` (dev server and nginx image).
- Training run completed and registered `surgecast_demand_15m` and `surgecast_demand_30m` version 1. Metrics live only in `results/train_metrics.json`.
- On this VM, after installing `libgomp1` in the image and bind-mounting the host artifact directory into the API container, `GET /api/health` returned `model_15` 1 and `model_30` 1 with Redis up. That was a health check, not a load test.

## Left

- MLflow recorded artifact paths as a host absolute directory (`/agent/repos/.../mlartifacts/...`). A container that does not mount that exact path cannot load the boosters. The mount is `HOST_ARTIFACT_DIR` in Compose. This still needs a portable artifact root.
- Live benchmarks were not run: producer rate, processor rate, feature-update latency, API p50/p95/p99, and drift-detection-to-recovered-model time. `python -m surgecast.bench.run` is the command. It was not executed.
- The drift service process was started, but a shock replay plus automatic retrain and API hot-swap was not demonstrated.
- Grafana was started with the provisioned dashboard. The dashboard was not opened or checked against live series.
- The map was not driven by a replay, so it has not been shown with live zone features.
- README resume bullets and `RESULTS.md` still need to be filled only from a real benchmark run.

## How to run what exists

From the repo root, with Docker and Python 3.12:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
make download-data
make mlflow
# other shell
export MLFLOW_TRACKING_URI=http://127.0.0.1:5055
make train
HOST_ARTIFACT_DIR="$PWD/mlartifacts" make up
make test
make replay SPEED=0 MAX_EVENTS=100000
cd frontend && npm install && npm run dev
```

`make up` expects `mlflow.db` and `mlartifacts/` from the training server. Stop the host MLflow process before Compose opens the same SQLite file.

On this VM, container-to-container traffic on the Compose bridge was dropped by iptables-legacy `FORWARD` policy DROP until this rule existed:

```bash
sudo iptables-legacy -I DOCKER-USER -s 172.18.0.0/16 -d 172.18.0.0/16 -j ACCEPT
```

Ports when Compose is up: API `8841`, map image `43124`, Vite dev map `43123`, Grafana `3333`, MLflow `5055`, Prometheus `9095`, Redpanda `19092`. Grafana local admin is `admin` / `admin`.

Trip parquet, `mlartifacts/`, `mlruns/`, `mlflow.db`, virtualenvs, and `frontend/node_modules/` are gitignored.
