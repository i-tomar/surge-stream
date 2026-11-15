"""Replay TLC yellow-taxi pickups into Kafka as a paced event stream."""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime

import orjson
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

from surgecast.config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, NY, RUNTIME_DIR, ensure_dirs
from surgecast.data.prepare import load_trips
from surgecast.metrics.prom import EVENTS_PRODUCED, serve_metrics


def ensure_topic(bootstrap: str, topic: str) -> None:
    admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="surgecast-replay")
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
    except TopicAlreadyExistsError:
        return
    except Exception as exc:
        if "already exists" in str(exc).lower() or "TopicExists" in str(exc):
            return
        raise
    finally:
        admin.close()


def _parse_local(value: str | None) -> int | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=NY)
    return int(parsed.timestamp())


def replay(
    *,
    speed: float = 0.0,
    max_events: int = 0,
    start: str | None = None,
    end: str | None = None,
    multiplier: float = 1.0,
    bootstrap: str = KAFKA_BOOTSTRAP,
    topic: str = KAFKA_TOPIC,
    metrics_port: int = 0,
) -> dict:
    """Publish trips in pickup-time order.

    ``speed`` is a multiplier on event time. ``0`` publishes as fast as the
    client can. ``multiplier`` repeats each trip so a demand shock can be
    replayed; ``3`` emits every trip three times at the same timestamp.
    ``max_events`` counts source trips, before the multiplier.
    """
    if metrics_port:
        serve_metrics(metrics_port)
    ensure_dirs()
    epochs, zones, stats = load_trips()
    start_epoch = _parse_local(start)
    end_epoch = _parse_local(end)
    if start_epoch is not None:
        keep = epochs >= start_epoch
        epochs, zones = epochs[keep], zones[keep]
    if end_epoch is not None:
        keep = epochs < end_epoch
        epochs, zones = epochs[keep], zones[keep]
    if max_events and len(epochs) > max_events:
        epochs = epochs[:max_events]
        zones = zones[:max_events]
    if len(epochs) == 0:
        raise RuntimeError("replay filter matched zero trips")

    copies = int(math.floor(multiplier))
    frac = float(multiplier) - copies
    if copies < 0:
        raise ValueError("multiplier must be >= 0")
    ensure_topic(bootstrap, topic)
    producer = KafkaProducer(
        bootstrap_servers=bootstrap,
        acks=1,
        linger_ms=5,
        batch_size=64 * 1024,
        value_serializer=orjson.dumps,
        retries=3,
    )
    started = time.perf_counter()
    origin_ts = int(epochs[0])
    emitted = 0
    # Deterministic fractional copies so a shock is reproducible.
    frac_tick = 0.0
    try:
        for index, (ts, zone) in enumerate(zip(epochs, zones, strict=True)):
            if speed > 0:
                target = (int(ts) - origin_ts) / speed
                delay = target - (time.perf_counter() - started)
                if delay > 0.002:
                    time.sleep(delay)
            n_copies = copies
            if frac:
                frac_tick += frac
                if frac_tick >= 1.0:
                    n_copies += 1
                    frac_tick -= 1.0
            payload_ts = int(ts)
            payload_zone = int(zone)
            for _ in range(n_copies):
                producer.send(topic, {"ts": payload_ts, "zone": payload_zone})
                emitted += 1
            if n_copies:
                EVENTS_PRODUCED.inc(n_copies)
            if index and index % 200_000 == 0:
                producer.flush()
        producer.flush()
    finally:
        producer.close()
    elapsed = time.perf_counter() - started
    # The periodic counter above is a coarse scrape aid. The file below is exact.
    report = {
        "source_events": int(len(epochs)),
        "emitted_events": int(emitted),
        "elapsed_sec": elapsed,
        "emitted_per_sec": emitted / elapsed if elapsed else None,
        "speed": speed,
        "multiplier": multiplier,
        "start": start,
        "end": end,
        "first_epoch": int(epochs[0]),
        "last_epoch": int(epochs[-1]),
        "bootstrap": bootstrap,
        "topic": topic,
        "load_stats": stats,
    }
    path = RUNTIME_DIR / "replay_stats.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"wrote": str(path), "emitted_per_sec": report["emitted_per_sec"], "emitted_events": emitted}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay TLC trips into Kafka")
    parser.add_argument("--speed", type=float, default=0.0, help="Event-time multiplier. 0 = no sleep.")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--start", default="", help="Inclusive local ISO time, e.g. 2024-01-20")
    parser.add_argument("--end", default="", help="Exclusive local ISO time")
    parser.add_argument("--multiplier", type=float, default=1.0, help="Demand-shock copies per trip")
    parser.add_argument("--metrics-port", type=int, default=8843)
    args = parser.parse_args()
    replay(
        speed=args.speed,
        max_events=args.max_events,
        start=args.start or None,
        end=args.end or None,
        multiplier=args.multiplier,
        metrics_port=args.metrics_port,
    )


if __name__ == "__main__":
    main()
