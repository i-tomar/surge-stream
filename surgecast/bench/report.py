"""Render RESULTS.md and the README measured section from a benchmark dict.

Missing measurements stay TBD. This module does not invent values.
"""

from __future__ import annotations

import json
from pathlib import Path


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "TBD"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _section_missing(title: str, payload: dict | None, command: str) -> str:
    if payload and payload.get("status") == "ok":
        return ""
    reason = (payload or {}).get("reason", "not run")
    return f"### {title}\n\nTBD. {reason}\n\nReproduce with:\n\n```bash\n{command}\n```\n"


def render_results(blob: dict) -> str:
    hw = blob.get("hardware") or {}
    lines = [
        "# Surgecast measurements",
        "",
        "Every number in this file was written by `python -m surgecast.bench.run` from a measurement dict.",
        "Raw values are in `results/benchmark.json`. Display rounding is applied only in this prose.",
        "",
        "## Environment",
        "",
        f"- Platform: {hw.get('platform', 'TBD')}",
        f"- Python: {hw.get('python', 'TBD')}",
        f"- CPU: {hw.get('cpu_model', 'TBD')} ({hw.get('cpu_count', 'TBD')} logical CPUs)",
        f"- MemTotal: {_fmt(hw.get('mem_total_kib'), 0)} KiB",
        f"- MemAvailable at report time: {_fmt(hw.get('mem_available_kib'), 0)} KiB",
        f"- Recorded at: {blob.get('generated_at', 'TBD')}",
        "",
    ]
    train = blob.get("train")
    if train and train.get("status") == "ok":
        body = train["report"]
        lgb = body["lightgbm"]
        lines += [
            "## Holdout forecast error",
            "",
            "LightGBM, time-based holdout. Test decisions start at "
            f"{body['splits']['test_start_local']}. "
            f"Rows: train {body['rows']['train']:,}, val {body['rows']['val']:,}, test {body['rows']['test']:,}.",
            "MAE / RMSE / MAPE below are on test rows with actual pickups > 0. "
            "MAPE is the mean of |error| / actual on those rows, in percent. "
            "Overall MAE includes zero-pickup zone-slots.",
            "",
            "| Horizon | Model | MAE (y>0) | RMSE (y>0) | MAPE % (y>0) | MAE (all rows) | n (y>0) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for horizon, key in (("15 min", "horizon_15"), ("30 min", "horizon_30")):
            block = lgb[key]
            rows = [("LightGBM", block["model"])]
            rows.append(("last value", block["baselines"]["last_value"]))
            rows.append(("seasonal naive (yesterday)", block["baselines"]["seasonal_naive_yesterday"]))
            for name, scores in rows:
                lines.append(
                    "| {h} | {n} | {mae} | {rmse} | {mape} | {allmae} | {npos} |".format(
                        h=horizon,
                        n=name,
                        mae=_fmt(scores.get("mae_y_gt_0")),
                        rmse=_fmt(scores.get("rmse_y_gt_0")),
                        mape=_fmt(scores.get("mape_pct_y_gt_0")),
                        allmae=_fmt(scores.get("mae")),
                        npos=_fmt(scores.get("n_positive"), 0),
                    )
                )
        torch_body = body.get("torch")
        if torch_body:
            lines += ["", "### PyTorch MLP (comparison, not served)", ""]
            for label, key in (("15 min", "horizon_15"), ("30 min", "horizon_30")):
                scores = torch_body[key]["scores"]
                lines.append(
                    f"- {label}: MAE (y>0) {_fmt(scores.get('mae_y_gt_0'))}, "
                    f"RMSE (y>0) {_fmt(scores.get('rmse_y_gt_0'))}, "
                    f"MAPE {_fmt(scores.get('mape_pct_y_gt_0'))}% "
                    f"over {scores.get('n_positive')} positive rows."
                )
            lines.append(f"- Fit time for both horizons: {_fmt(torch_body.get('seconds'), 2)} s.")
        elif body.get("torch_error"):
            lines += ["", f"PyTorch comparison did not finish: {body['torch_error']}", ""]
        lines += [
            "",
            f"LightGBM fit seconds: 15m {_fmt(lgb.get('fit_seconds_15'), 2)}, "
            f"30m {_fmt(lgb.get('fit_seconds_30'), 2)}.",
            f"Champion versions at train time: 15m v{lgb.get('version_15')}, 30m v{lgb.get('version_30')}.",
            f"Best iteration: 15m {lgb.get('best_iteration_15')}, 30m {lgb.get('best_iteration_30')} "
            f"(cap {lgb.get('num_boost_round_cap')}).",
            f"Trips kept: {body.get('counts_meta', {}).get('n_kept')}.",
            "",
        ]
    else:
        lines.append(_section_missing("Holdout forecast error", train, "make train"))

    for title, key, command in (
        ("API latency", "api_latency", "python -m surgecast.bench.run"),
        ("Throughput", "throughput", "make replay SPEED=0 MAX_EVENTS=100000"),
        ("Feature latency", "feature_latency", "python -m surgecast.bench.run"),
        ("Drift recovery", "drift", "python -m surgecast.bench.run"),
    ):
        payload = blob.get(key)
        if not payload or payload.get("status") != "ok":
            lines.append(_section_missing(title, payload, command))
            continue
        lines.append(f"## {title}")
        lines.append("")
        lines.append(payload["markdown"].rstrip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_measured_block(blob: dict) -> str:
    """Short README block. Only includes facts present in the blob."""
    parts = ["<!-- MEASURED:START -->", "", "## Resume bullets", ""]
    hw = blob.get("hardware") or {}
    parts.append(
        f"Measured on {hw.get('cpu_model', 'this machine')} "
        f"({hw.get('cpu_count', '?')} CPUs, MemTotal {_fmt(hw.get('mem_total_kib'), 0)} KiB), "
        f"Python {hw.get('python', '?')}. Source: `results/benchmark.json`, generated {blob.get('generated_at', '')}."
    )
    parts.append("")
    train = blob.get("train") or {}
    if train.get("status") == "ok":
        lgb = train["report"]["lightgbm"]
        m15 = lgb["horizon_15"]["model"]
        b15 = lgb["horizon_15"]["baselines"]["last_value"]
        s15 = lgb["horizon_15"]["baselines"]["seasonal_naive_yesterday"]
        m30 = lgb["horizon_30"]["model"]
        parts.append(
            f"- 15-minute holdout, rows with y>0 (n={m15['n_positive']}): "
            f"LightGBM MAE {_fmt(m15['mae_y_gt_0'])}, RMSE {_fmt(m15['rmse_y_gt_0'])}, "
            f"MAPE {_fmt(m15['mape_pct_y_gt_0'])}%. "
            f"Last-value MAE {_fmt(b15['mae_y_gt_0'])}. "
            f"Seasonal-naive MAE {_fmt(s15['mae_y_gt_0'])}."
        )
        parts.append(
            f"- 30-minute holdout, rows with y>0 (n={m30['n_positive']}): "
            f"LightGBM MAE {_fmt(m30['mae_y_gt_0'])}, RMSE {_fmt(m30['rmse_y_gt_0'])}, "
            f"MAPE {_fmt(m30['mape_pct_y_gt_0'])}%."
        )
    api = blob.get("api_latency") or {}
    if api.get("status") == "ok":
        parts.append(
            f"- API single-zone predict, concurrency {api['concurrency']}: "
            f"p50 {_fmt(api['p50_ms'], 2)} ms, p95 {_fmt(api['p95_ms'], 2)} ms, "
            f"p99 {_fmt(api['p99_ms'], 2)} ms (n={api['n']}, successes={api['successes']})."
        )
    thru = blob.get("throughput") or {}
    if thru.get("status") == "ok":
        parts.append(
            f"- Replay producer: {_fmt(thru.get('produced_per_sec'), 1)} events/s "
            f"over {thru.get('produced_events')} events."
        )
        if thru.get("consumed_per_sec") is not None:
            parts.append(f"- Feature processor: {_fmt(thru.get('consumed_per_sec'), 1)} events/s over the same window.")
    feat = blob.get("feature_latency") or {}
    if feat.get("status") == "ok":
        parts.append(
            f"- End-to-end feature latency (produce ack to Redis update): "
            f"p50 {_fmt(feat['p50_ms'], 2)} ms, p95 {_fmt(feat['p95_ms'], 2)} ms (n={feat['n']})."
        )
    drift = blob.get("drift") or {}
    if drift.get("status") == "ok" and drift.get("detect_to_reload_seconds") is not None:
        parts.append(
            f"- Drift detection to reloaded model: {_fmt(drift['detect_to_reload_seconds'], 2)} s "
            f"({drift.get('mode')})."
        )
    if len(parts) <= 3:
        parts.append("No measurements have been recorded yet. Run `python -m surgecast.bench.run`.")
    parts += ["", "Resume bullets use only the lines above. Full tables are in `RESULTS.md`.", "", "<!-- MEASURED:END -->"]
    return "\n".join(parts) + "\n"


def write_reports(blob: dict, root: Path) -> None:
    results = root / "RESULTS.md"
    results.write_text(render_results(blob))
    readme = root / "README.md"
    if not readme.exists():
        return
    text = readme.read_text()
    start = "<!-- MEASURED:START -->"
    end = "<!-- MEASURED:END -->"
    block = render_measured_block(blob).rstrip() + "\n"
    if start in text and end in text:
        pre, _, rest = text.partition(start)
        _, _, post = rest.partition(end)
        readme.write_text(pre + block + post.lstrip("\n"))
    (root / "results" / "benchmark.json").write_text(json.dumps(blob, indent=2, default=str))
