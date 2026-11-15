"""Prometheus metrics shared by the API, processor, replay, and drift service."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

EVENTS_PRODUCED = Counter(
    "surgecast_events_produced_total",
    "Trip events accepted by the replay producer",
)
EVENTS_CONSUMED = Counter(
    "surgecast_events_consumed_total",
    "Trip events applied by the feature processor",
)
LATE_EVENTS = Counter(
    "surgecast_late_events_total",
    "Events whose bucket is behind the event-time watermark",
)
CONSUMER_LAG = Gauge(
    "surgecast_consumer_lag",
    "Sum of Kafka end offsets minus the consumer position",
)
FEATURE_LATENCY = Histogram(
    "surgecast_feature_update_latency_seconds",
    "Wall time from reading a bucket-advancing event to finishing the Redis feature write",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
API_LATENCY = Histogram(
    "surgecast_api_request_latency_seconds",
    "FastAPI handler latency, including the Redis read and model predict",
    labelnames=("endpoint",),
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2),
)
PREDICTION_ABS_ERROR = Gauge(
    "surgecast_prediction_abs_error",
    "Mean absolute error of the champion model on the recent labeled sample",
    labelnames=("horizon",),
)
DRIFT_SCORE = Gauge(
    "surgecast_drift_score",
    "Share of watched columns Evidently marked as drifted",
)
DRIFT_DETECTED = Gauge(
    "surgecast_drift_detected",
    "1 when the latest drift check requested a retrain",
)
MODEL_VERSION = Gauge(
    "surgecast_model_version",
    "Champion model version",
    labelnames=("horizon",),
)
WATERMARK = Gauge(
    "surgecast_watermark_epoch",
    "Event-time watermark as unix seconds",
)


def serve_metrics(port: int) -> None:
    start_http_server(port)
