"""Demand-shock replay.

Emits ``multiplier`` copies of each trip in a local-time window. Run it against
a window the main replay has not published yet, so the extra pickups land in
open buckets and move the feature store. Re-emitting an already consumed past
window only increments in-memory counts behind the watermark, which does not
refresh Redis.
"""

from __future__ import annotations

import argparse

import redis

from surgecast.config import REDIS_URL
from surgecast.streaming.replay import replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject a multiplicative demand shock")
    parser.add_argument("--multiplier", type=float, default=3.0)
    parser.add_argument("--start", required=True, help="Inclusive local ISO time")
    parser.add_argument("--end", required=True, help="Exclusive local ISO time")
    parser.add_argument("--speed", type=float, default=0.0)
    args = parser.parse_args()
    client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    client.set("meta:shock_multiplier", str(args.multiplier))
    client.set("meta:shock_start", args.start)
    client.set("meta:shock_end", args.end)
    replay(speed=args.speed, start=args.start, end=args.end, multiplier=args.multiplier, metrics_port=0)


if __name__ == "__main__":
    main()
