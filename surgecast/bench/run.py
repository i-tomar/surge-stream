"""Measure what this machine can actually run and write RESULTS.md.

Live Kafka / Redis / API sections are recorded only when those processes accept
a connection. Otherwise the section stays TBD with the command that would
produce it.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from surgecast.bench.hardware import hardware
from surgecast.bench.report import write_reports
from surgecast.config import (
    KAFKA_BOOTSTRAP,
    KAFKA_TOPIC,
    N_ZONES,
    ORIGIN_EPOCH,
    REDIS_URL,
    RESULTS_DIR,
    ROOT,
    RUNTIME_DIR,
    TEST_START_DT,
    ensure_dirs,
)
from surgecast.features.compute import FEATURE_COLUMNS, bucket_of, epoch_of_bucket

API = os.environ.get("SURGECAST_API", "http://127.0.0.1:8841")


def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _load_train() -> dict:
    path = RESULTS_DIR / "train_metrics.json"
    if not path.exists():
        return {"status": "not_measured", "reason": f"{path} does not exist. Train first."}
    return {"status": "ok", "report": json.loads(path.read_text())}


def _seed_or_existing() -> str:
    import redis

    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    if client.hlen("zone:161"):
        return "existing Redis hashes"
    pipe = client.pipeline(transaction=False)
    for zone in range(1, N_ZONES + 1):
        mapping = {column: "1" for column in FEATURE_COLUMNS if column != "zone_id"}
        mapping["zone_id"] = str(zone)
        mapping["pu_15m"] = "12"
        mapping["pu_30m"] = "20"
        mapping["bucket"] = "2500"
        mapping["decision_epoch"] = str(ORIGIN_EPOCH + 2500 * 300)
        pipe.hset(f"zone:{zone}", mapping=mapping)
    pipe.execute()
    return "benchmark-seeded Redis hashes (the feature store was empty)"


async def _latency(path_builder, n: int, concurrency: int, warmup: int) -> dict:
    import httpx

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    samples: list[float] = []
    codes: dict[str, int] = {}
    async with httpx.AsyncClient(base_url=API, timeout=10.0, limits=limits) as client:
        for _ in range(warmup):
            await client.get(path_builder())
        sem = asyncio.Semaphore(concurrency)

        async def one() -> None:
            async with sem:
                started = time.perf_counter()
                response = await client.get(path_builder())
                elapsed = time.perf_counter() - started
            samples.append(elapsed)
            codes[str(response.status_code)] = codes.get(str(response.status_code), 0) + 1

        await asyncio.gather(*[one() for _ in range(n)])
    arr = np.array(samples, dtype=np.float64)
    ok = codes.get("200", 0)
    return {
        "n": n,
        "concurrency": concurrency,
        "warmup": warmup,
        "successes": ok,
        "status_codes": codes,
        "p50_ms": float(np.percentile(arr, 50) * 1000) if len(arr) else None,
        "p95_ms": float(np.percentile(arr, 95) * 1000) if len(arr) else None,
        "p99_ms": float(np.percentile(arr, 99) * 1000) if len(arr) else None,
        "mean_ms": float(arr.mean() * 1000) if len(arr) else None,
    }


def measure_api() -> dict:
    host, port = "127.0.0.1", int(API.rsplit(":", 1)[-1])
    if not _port_open(host, port):
        return {"status": "not_measured", "reason": f"API is not accepting connections on {API}."}
    if not _port_open("127.0.0.1", 6379):
        return {"status": "not_measured", "reason": "Redis is not accepting connections on 127.0.0.1:6379."}
    try:
        source = _seed_or_existing()
        rng = random.Random(7)

        def predict_path() -> str:
            return f"/api/predict/{rng.randint(1, N_ZONES)}?horizon=15"

        single = asyncio.run(_latency(predict_path, n=400, concurrency=1, warmup=40))
        loaded = asyncio.run(_latency(predict_path, n=1500, concurrency=32, warmup=80))
        zones = asyncio.run(_latency(lambda: "/api/zones", n=200, concurrency=8, warmup=10))
    except Exception as exc:
        return {"status": "not_measured", "reason": f"{type(exc).__name__}: {exc}"}
    if loaded["successes"] == 0:
        return {
            "status": "not_measured",
            "reason": f"No HTTP 200 from /api/predict. Status codes: {loaded['status_codes']}.",
        }
    markdown = "\n".join(
        [
            f"Feature source: {source}.",
            "Clock is client-side `time.perf_counter` around `httpx` GET, after a warmup that is excluded.",
            "",
            "| Endpoint | Concurrency | n | successes | p50 ms | p95 ms | p99 ms | mean ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            _lat_row("/api/predict/{zone}", single),
            _lat_row("/api/predict/{zone}", loaded),
            _lat_row("/api/zones", zones),
        ]
    )
    return {"status": "ok", "markdown": markdown, **loaded, "single": single, "zones": zones, "feature_source": source}


def _lat_row(name: str, row: dict) -> str:
    return (
        f"| {name} | {row['concurrency']} | {row['n']} | {row['successes']} | "
        f"{row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} | {row['mean_ms']:.3f} |"
    )


def measure_throughput() -> dict:
    host, port_s = KAFKA_BOOTSTRAP.split(":")
    if not _port_open(host, int(port_s)):
        return {"status": "not_measured", "reason": f"Kafka API is not accepting connections on {KAFKA_BOOTSTRAP}."}
    before = _read_json(RUNTIME_DIR / "processor_stats.json")
    started = time.perf_counter()
    proc = subprocess.run(
        [
            os.environ.get("SURGECAST_PY", str(ROOT / ".venv/bin/python")),
            "-m",
            "surgecast.streaming.replay",
            "--speed",
            "0",
            "--max-events",
            os.environ.get("BENCH_EVENTS", "100000"),
            "--metrics-port",
            "0",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    wall = time.perf_counter() - started
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-800:]
        return {"status": "not_measured", "reason": f"replay exited {proc.returncode}: {tail}"}
    produced = _read_json(RUNTIME_DIR / "replay_stats.json") or {}
    deadline = time.perf_counter() + 90
    after = _read_json(RUNTIME_DIR / "processor_stats.json")
    while time.perf_counter() < deadline:
        current = _read_json(RUNTIME_DIR / "processor_stats.json")
        if current and before and current.get("events", 0) >= before.get("events", 0) + produced.get("emitted_events", 0):
            after = current
            break
        if current:
            after = current
        time.sleep(1)
    consumed = None
    consumed_per_sec = None
    if before and after:
        delta_events = after.get("events", 0) - before.get("events", 0)
        if delta_events > 0:
            consumed = delta_events
            consumed_per_sec = delta_events / max(time.perf_counter() - started, 1e-6)
    markdown = "\n".join(
        [
            "Producer timing starts after the parquet is loaded and sorted (`surgecast.streaming.replay`).",
            f"Emitted events: {produced.get('emitted_events')}. Elapsed inside the producer: {produced.get('elapsed_sec')} s.",
            f"Producer rate: {produced.get('emitted_per_sec')} events/s.",
            f"Subprocess wall time, including Python startup: {wall:.3f} s.",
            f"Processor events applied during the window: {consumed}.",
            f"Processor rate over the window (including catch-up wait): {consumed_per_sec}.",
            f"Processor lag field after the run: {(after or {}).get('lag')}. Late events: {(after or {}).get('late_events')}.",
        ]
    )
    return {
        "status": "ok",
        "markdown": markdown,
        "produced_events": produced.get("emitted_events"),
        "produced_per_sec": produced.get("emitted_per_sec"),
        "consumed_events": consumed,
        "consumed_per_sec": consumed_per_sec,
        "producer_elapsed_sec": produced.get("elapsed_sec"),
        "late_events": (after or {}).get("late_events"),
        "lag": (after or {}).get("lag"),
    }


def measure_feature_latency() -> dict:
    host, port_s = KAFKA_BOOTSTRAP.split(":")
    if not _port_open(host, int(port_s)) or not _port_open("127.0.0.1", 6379):
        return {"status": "not_measured", "reason": "Kafka or Redis is not up, so end-to-end feature latency was not measured."}
    import orjson
    import redis
    from kafka import KafkaProducer

    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        acks=1,
        linger_ms=0,
        value_serializer=orjson.dumps,
    )
    try:
        from surgecast.streaming.replay import ensure_topic

        ensure_topic(KAFKA_BOOTSTRAP, KAFKA_TOPIC)
    except Exception as exc:
        producer.close()
        return {"status": "not_measured", "reason": f"could not ensure topic: {exc}"}
    current = client.get("meta:bucket")
    base = int(current) + 4 if current else 4000
    samples = []
    try:
        for trial in range(12):
            target = base + trial * 3
            bucket_epoch = epoch_of_bucket(target - 1, ORIGIN_EPOCH) + 10
            for _ in range(6):
                producer.send(KAFKA_TOPIC, {"ts": bucket_epoch, "zone": 42})
            producer.flush()
            closer = epoch_of_bucket(target, ORIGIN_EPOCH) + 10
            started = time.perf_counter()
            producer.send(KAFKA_TOPIC, {"ts": closer, "zone": 7}).get(timeout=10)
            deadline = started + 5
            seen = None
            while time.perf_counter() < deadline:
                raw = client.hget("zone:42", "bucket")
                if raw is not None and int(float(raw)) >= target:
                    seen = (time.perf_counter() - started) * 1000
                    break
                time.sleep(0.002)
            if seen is not None:
                samples.append(seen)
    finally:
        producer.close()
    if len(samples) < 5:
        return {
            "status": "not_measured",
            "reason": f"Only {len(samples)} of 12 feature-latency trials observed a Redis update. Is the processor running?",
        }
    arr = np.array(samples)
    result = {
        "status": "ok",
        "n": int(len(samples)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, min(99, 100))),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }
    result["markdown"] = "\n".join(
        [
            "Each trial writes 6 pickups for zone 42 into a fresh future bucket, then sends one later event that closes it.",
            "The clock starts before the closing send and stops when Redis `zone:42` bucket reaches that decision.",
            "This includes the producer ack, Kafka, the processor, and Redis. It does not include waiting out a real 5 minutes;",
            "the bucket closes when the next bucket's event arrives.",
            "",
            f"n={result['n']} p50={result['p50_ms']:.3f} ms p95={result['p95_ms']:.3f} ms "
            f"min={result['min_ms']:.3f} ms max={result['max_ms']:.3f} ms.",
        ]
    )
    return result


def measure_drift() -> dict:
    metrics_path = RESULTS_DIR / "train_metrics.json"
    counts_path = ROOT / "data" / "processed" / "bucket_counts.npz"
    reference_path = ROOT / "data" / "processed" / "reference_features.parquet"
    if not metrics_path.exists() or not counts_path.exists() or not reference_path.exists():
        return {
            "status": "not_measured",
            "reason": "Need train artifacts (counts, reference features, train_metrics.json).",
        }
    import pandas as pd

    from surgecast.data.prepare import load_counts
    from surgecast.drift.detect import detect_drift
    from surgecast.drift.retrain import retrain_from_counts
    from surgecast.serving.model_loader import load_champions
    from surgecast.training.dataset import build_frame, matrix
    from surgecast.training.metrics import clip_prediction, regression_scores

    counts, _ = load_counts()
    shock_bucket = bucket_of(int(TEST_START_DT.timestamp()), ORIGIN_EPOCH)
    shocked = counts.copy()
    shocked[:, shock_bucket:] = np.clip(
        np.rint(shocked[:, shock_bucket:].astype(np.float64) * 3),
        0,
        np.iinfo(np.int32).max,
    ).astype(np.int32)
    frame = build_frame(shocked)
    early = frame[(frame["bucket"] >= shock_bucket) & (frame["bucket"] < shock_bucket + 12 * 12)]
    if len(early) < 100:
        return {"status": "not_measured", "reason": f"Shock window produced only {len(early)} rows."}
    reference = pd.read_parquet(reference_path)
    started = time.perf_counter()
    report = detect_drift(reference, early)
    detect_s = time.perf_counter() - started
    if not report.dataset_drift:
        return {
            "status": "not_measured",
            "reason": (
                f"3x shock on buckets from the test start did not trip Evidently "
                f"(share_drifted={report.share_drifted}, detect_seconds={report.seconds})."
            ),
            "share_drifted": report.share_drifted,
            "detect_seconds": report.seconds,
        }
    original = load_champions()
    _, test_df = _train_and_test(frame)
    base_scores = _score_pair(original, test_df, clip_prediction, regression_scores, matrix)
    temp = ROOT / "results" / "drift-mlruns"
    temp.mkdir(parents=True, exist_ok=True)
    previous = os.environ.get("MLFLOW_TRACKING_URI")
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{temp}"
    retrain_started = time.perf_counter()
    try:
        promoted = retrain_from_counts(shocked, reason="benchmark-3x-shock", num_boost_round=200)
        reloaded = load_champions(f"file:{temp}")
    finally:
        if previous is None:
            os.environ.pop("MLFLOW_TRACKING_URI", None)
        else:
            os.environ["MLFLOW_TRACKING_URI"] = previous
    recover_s = time.perf_counter() - retrain_started
    total_s = time.perf_counter() - started
    new_scores = _score_pair(reloaded, test_df, clip_prediction, regression_scores, matrix)
    result = {
        "status": "ok",
        "mode": "in-process detect + retrain + reload (not the docker API poll)",
        "dataset_drift": True,
        "share_drifted": report.share_drifted,
        "detect_seconds": detect_s,
        "retrain_and_reload_seconds": recover_s,
        "detect_to_reload_seconds": total_s,
        "shock": "counts from the test-start bucket onward multiplied by 3",
        "original_on_shocked_test": base_scores,
        "retrained_on_shocked_test": new_scores,
        "promoted": {"version_15": promoted["version_15"], "version_30": promoted["version_30"], "retrain_seconds": promoted["seconds"]},
    }
    result["markdown"] = _drift_markdown(result)
    return result


def _train_and_test(frame):
    from surgecast.training.dataset import time_splits

    _train_df, _val_df, test_df = time_splits(frame)
    return _train_df, test_df


def _score_pair(champions, test_df, clip_prediction, regression_scores, matrix):
    features, y15, y30 = matrix(test_df)
    _, model_15, _, model_30 = champions
    return {
        "horizon_15": regression_scores(y15, clip_prediction(model_15.predict(features))),
        "horizon_30": regression_scores(y30, clip_prediction(model_30.predict(features))),
    }


def _drift_markdown(result: dict) -> str:
    def line(label, scores):
        s15 = scores["horizon_15"]
        s30 = scores["horizon_30"]
        return (
            f"- {label}: 15m MAE(y>0)={s15['mae_y_gt_0']:.4f} MAPE={s15['mape_pct_y_gt_0']:.4f}% "
            f"(n={s15['n_positive']}); 30m MAE(y>0)={s30['mae_y_gt_0']:.4f} "
            f"MAPE={s30['mape_pct_y_gt_0']:.4f}% (n={s30['n_positive']})."
        )

    return "\n".join(
        [
            result["mode"] + ".",
            result["shock"] + ".",
            f"Evidently dataset_drift={result['dataset_drift']} share_drifted={result['share_drifted']:.4f} "
            f"detect_seconds={result['detect_seconds']:.4f}.",
            f"Retrain + registry alias + model reload: {result['retrain_and_reload_seconds']:.4f} s.",
            f"Detect start to reloaded model: {result['detect_to_reload_seconds']:.4f} s.",
            "Scores are on the shocked test split (decisions from 2024-01-28). "
            "The original champion was trained on unshocked data. The retrained model was fit on the shocked counts.",
            line("Original champion on shocked test", result["original_on_shocked_test"]),
            line("Retrained champion on shocked test", result["retrained_on_shocked_test"]),
        ]
    )


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def main() -> None:
    ensure_dirs()
    print("benchmark: hardware and train metrics", flush=True)
    blob = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware(),
        "train": _load_train(),
        "command": "python -m surgecast.bench.run",
    }
    print("benchmark: api", flush=True)
    try:
        blob["api_latency"] = measure_api()
    except Exception as exc:
        blob["api_latency"] = {"status": "not_measured", "reason": f"{type(exc).__name__}: {exc}"}
    print("benchmark: throughput", flush=True)
    try:
        blob["throughput"] = measure_throughput()
    except Exception as exc:
        blob["throughput"] = {"status": "not_measured", "reason": f"{type(exc).__name__}: {exc}"}
    print("benchmark: feature latency", flush=True)
    try:
        blob["feature_latency"] = measure_feature_latency()
    except Exception as exc:
        blob["feature_latency"] = {"status": "not_measured", "reason": f"{type(exc).__name__}: {exc}"}
    print("benchmark: drift", flush=True)
    try:
        blob["drift"] = measure_drift()
    except Exception as exc:
        blob["drift"] = {"status": "not_measured", "reason": f"{type(exc).__name__}: {exc}"}
    write_reports(blob, ROOT)
    print(
        json.dumps(
            {
                key: blob[key].get("status") if isinstance(blob[key], dict) else "ok"
                for key in ("train", "api_latency", "throughput", "feature_latency", "drift")
            }
        )
    )


if __name__ == "__main__":
    main()
