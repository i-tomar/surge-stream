"""Download the public TLC files used by the pipeline."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from surgecast.config import LOOKUP_URL, RAW_DIR, TRIP_URL, ensure_dirs


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    print(f"get {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"wrote {dest} ({dest.stat().st_size} bytes)")


def main() -> None:
    ensure_dirs()
    _fetch(TRIP_URL, RAW_DIR / "yellow_tripdata_2024-01.parquet")
    _fetch(LOOKUP_URL, RAW_DIR / "taxi_zone_lookup.csv")
    print("Taxi-zone GeoJSON is checked in at frontend/public/taxi_zones.geojson")


if __name__ == "__main__":
    main()
