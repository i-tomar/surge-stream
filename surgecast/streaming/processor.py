"""Consume trip events and publish rolling features to Redis.

Events are processed in Kafka order on a single partition. The watermark is
the latest event time. A decision at bucket ``t`` is written once an event in
bucket ``t`` or later arrives, so the feature vector only contains counts from
buckets ``< t``. That is the same index convention as ``feature_rows``.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time

import numpy as np
import orjson
import redis
from kafka import KafkaConsumer, TopicPartition

from surgecast.config import (
    KAFKA_BOOTSTRAP,
    KAFKA_TOPIC,
    N_ZONES,
    ORIGIN_EPOCH,
    REDIS_URL,
    RUNTIME_DIR,
    WEEK_BUCKETS,
    ensure_dirs,
)
from surgecast.features.compute import FEATURE_COLUMNS, bucket_of, epoch_of_bucket, feature_rows
from surgecast.metrics.prom import (
    CONSUMER_LAG,
    EVENTS_CONSUMED,
    FEATURE_LATENCY,
    LATE_EVENTS,
    WATERMARK,
    serve_metrics,
)

logger = logging.getLogger("surgecast.processor")

SAMPLE_ZONES = 48
RECENT_LIMIT = 3000


class CountStore:
    def __init__(self) -> None:
        self.counts = np.zeros((N_ZONES + 1, 16), dtype=np.int32)

    def add(self, zone: int, bucket: int, n: int = 1) -> None:
        if bucket < 0 or zone < 1 or zone > N_ZONES:
            return
        if bucket >= self.counts.shape[1]:
            new_len = max(bucket + 8, self.counts.shape[1] * 2)
            grown = np.zeros((self.counts.shape[0], new_len), dtype=np.int32)
            grown[:, : self.counts.shape[1]] = self.counts
            self.counts = grown
        self.counts[zone, bucket] += n

    def active_zones(self) -> np.ndarray:
        totals = self.counts.sum(axis=1)
        return np.flatnonzero(totals > 0)


def _encode(row: dict) -> dict[str, str]:
    encoded = {}
    for key, value in row.items():
        if value is None or key in {"y15", "y30"}:
            continue
        if key in {"zone_id", "bucket", "decision_epoch"}:
            encoded[key] = str(int(value))
        else:
            encoded[key] = format(float(value), ".8g")
    return encoded


def _jsonable(row: dict, keys: list[str], extra: dict | None = None) -> bytes:
    body: dict = {}
    for key in keys:
        value = row[key]
        body[key] = int(value) if key == "zone_id" else float(value)
    for key, value in (extra or {}).items():
        body[key] = None if value is None else float(value)
    return orjson.dumps(body)


def _sample_ids(bucket: int, available: np.ndarray) -> list[int]:
    if len(available) == 0:
        return []
    if len(available) <= SAMPLE_ZONES:
        return [int(z) for z in available]
    start = (bucket * SAMPLE_ZONES) % len(available)
    picked = [available[(start + i) % len(available)] for i in range(SAMPLE_ZONES)]
    return [int(z) for z in picked]


class FeatureProcessor:
    def __init__(self, redis_client: redis.Redis, origin_epoch: int = ORIGIN_EPOCH) -> None:
        self.redis = redis_client
        self.origin = origin_epoch
        self.store = CountStore()
        self.max_bucket = -1
        self.events = 0
        self.flushes = 0
        self.late = 0
        self.started = time.perf_counter()
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def handle_event(self, ts: int, zone: int) -> int | None:
        bucket = bucket_of(ts, self.origin)
        if bucket < 0 or zone < 1 or zone > N_ZONES:
            return None
        if self.max_bucket >= 0 and bucket < self.max_bucket:
            self.late += 1
            LATE_EVENTS.inc()
        self.store.add(zone, bucket)
        self.events += 1
        EVENTS_CONSUMED.inc()
        if bucket > self.max_bucket:
            previous = self.max_bucket
            self.max_bucket = bucket
            return previous
        return None

    def flush_range(self, previous_max: int, received_at: float) -> None:
        """Publish features for decisions ``previous_max+1 .. max_bucket``."""
        new_max = self.max_bucket
        if new_max <= previous_max:
            return
        decisions = list(range(previous_max + 1, new_max + 1))
        sampled = decisions
        if len(decisions) > 8:
            step = max(1, len(decisions) // 8)
            sampled = decisions[::step]
            if sampled[-1] != new_max:
                sampled.append(new_max)
        for decision in sampled:
            self._push_samples(decision)
        self._write_latest(new_max)
        FEATURE_LATENCY.observe(time.perf_counter() - received_at)
        self.flushes += 1

    def _rows(self, decision: int, zones: list[int]) -> list[dict]:
        rows = []
        for zone in zones:
            columns = feature_rows(self.store.counts[zone], np.asarray([decision]), zone, self.origin)
            row = {key: columns[key][0] for key in FEATURE_COLUMNS}
            row["bucket"] = int(columns["bucket"][0])
            row["decision_epoch"] = int(columns["decision_epoch"][0])
            row["y15"] = float(columns["y15"][0]) if decision + 3 <= self.max_bucket else None
            row["y30"] = float(columns["y30"][0]) if decision + 6 <= self.max_bucket else None
            rows.append(row)
        return rows

    def _push_samples(self, decision: int) -> None:
        zones = _sample_ids(decision, self.store.active_zones())
        if not zones or decision < WEEK_BUCKETS:
            return
        # Labels for an earlier decision are known once this decision is reached.
        labeled_decision = decision - 3
        pipe = self.redis.pipeline(transaction=False)
        for row in self._rows(decision, zones):
            pipe.lpush("features:recent", _jsonable(row, FEATURE_COLUMNS))
        if labeled_decision >= WEEK_BUCKETS:
            for row in self._rows(labeled_decision, _sample_ids(labeled_decision, self.store.active_zones())):
                if row["y15"] is None:
                    continue
                pipe.lpush(
                    "labeled:recent",
                    _jsonable(row, FEATURE_COLUMNS, {"y15": row["y15"], "y30": row["y30"]}),
                )
        pipe.ltrim("features:recent", 0, RECENT_LIMIT - 1)
        pipe.ltrim("labeled:recent", 0, RECENT_LIMIT - 1)
        pipe.execute()

    def _write_latest(self, decision: int) -> None:
        zones = [int(z) for z in self.store.active_zones()]
        pipe = self.redis.pipeline(transaction=False)
        epoch = epoch_of_bucket(decision, self.origin)
        for row in self._rows(decision, zones):
            zone = int(row["zone_id"])
            mapping = _encode(row)
            mapping.pop("y15", None)
            mapping.pop("y30", None)
            pipe.hset(f"zone:{zone}", mapping=mapping)
        pipe.set("meta:watermark", str(epoch))
        pipe.set("meta:bucket", str(decision))
        pipe.set("meta:zones", str(len(zones)))
        pipe.execute()
        WATERMARK.set(epoch)

    def save_counts(self) -> None:
        path = RUNTIME_DIR / "live_counts.npz"
        tmp = RUNTIME_DIR / "live_counts_tmp.npz"
        width = max(self.max_bucket + 1, 1)
        np.savez_compressed(tmp, counts=self.store.counts[:, :width])
        os.replace(tmp, path)

    def write_stats(self, lag: float | None = None) -> None:
        elapsed = time.perf_counter() - self.started
        report = {
            "events": self.events,
            "late_events": self.late,
            "flushes": self.flushes,
            "max_bucket": self.max_bucket,
            "elapsed_sec": elapsed,
            "events_per_sec": self.events / elapsed if elapsed else None,
            "lag": lag,
            "watermark_epoch": epoch_of_bucket(self.max_bucket, self.origin) if self.max_bucket >= 0 else None,
        }
        (RUNTIME_DIR / "processor_stats.json").write_text(json.dumps(report))


def _lag(consumer: KafkaConsumer, topic: str) -> float:
    partitions = consumer.partitions_for_topic(topic) or set()
    if not partitions:
        return 0.0
    tps = [TopicPartition(topic, partition) for partition in partitions]
    try:
        ends = consumer.end_offsets(tps)
    except Exception:
        return 0.0
    total = 0
    for tp in tps:
        position = consumer.position(tp)
        total += max(0, int(ends[tp]) - int(position))
    return float(total)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ensure_dirs()
    port = int(os.environ.get("METRICS_PORT", "8842"))
    serve_metrics(port)
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    processor = FeatureProcessor(client)
    signal.signal(signal.SIGTERM, lambda *_: processor.stop())
    signal.signal(signal.SIGINT, lambda *_: processor.stop())
    consumer = None
    while consumer is None and not processor._stop:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP,
                group_id="surgecast-features",
                enable_auto_commit=True,
                auto_offset_reset="earliest",
                max_poll_records=4000,
            )
        except Exception as exc:
            logger.warning("kafka not ready: %s", exc)
            time.sleep(2)
    assert consumer is not None
    logger.info("consuming %s from %s", KAFKA_TOPIC, KAFKA_BOOTSTRAP)
    last_stats = 0.0
    last_save = 0.0
    try:
        while not processor._stop:
            batches = consumer.poll(timeout_ms=500, max_records=4000)
            received_at = time.perf_counter()
            previous = processor.max_bucket
            advanced = False
            for records in batches.values():
                for record in records:
                    try:
                        payload = record.value if isinstance(record.value, dict) else orjson.loads(record.value)
                        ts = int(payload["ts"])
                        zone = int(payload["zone"])
                    except Exception:
                        continue
                    if processor.handle_event(ts, zone) is not None:
                        advanced = True
            if advanced:
                processor.flush_range(previous, received_at)
            now = time.perf_counter()
            if now - last_stats > 2:
                lag = _lag(consumer, KAFKA_TOPIC)
                CONSUMER_LAG.set(lag)
                processor.write_stats(lag)
                last_stats = now
            if now - last_save > 5 and processor.max_bucket >= 0:
                processor.save_counts()
                last_save = now
    finally:
        processor.write_stats()
        processor.save_counts()
        consumer.close()


if __name__ == "__main__":
    run()
