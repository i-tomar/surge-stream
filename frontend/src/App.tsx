import { useEffect, useMemo, useState } from "react";
import { GeoJSON, MapContainer, TileLayer } from "react-leaflet";
import type { Feature, FeatureCollection } from "geojson";
import type { Layer } from "leaflet";

type Horizon = 15 | 30;

type Zone = {
  zone_id: number;
  name: string;
  borough: string;
  prediction_15: number | null;
  prediction_30: number | null;
  pu_15m: number;
  pu_60m: number;
};

type Payload = {
  model_15: number | null;
  model_30: number | null;
  event_time: string | null;
  zones: Zone[];
};

function mix(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const stops: [number, number, number][] = [
    [18, 48, 64],
    [201, 132, 42],
    [255, 75, 50],
  ];
  const x = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const channel = (k: number) => Math.round(stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f);
  return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
}

function valueOf(zone: Zone, horizon: Horizon): number | null {
  return horizon === 15 ? zone.prediction_15 : zone.prediction_30;
}

function fmt(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

export default function App() {
  const [geo, setGeo] = useState<FeatureCollection | null>(null);
  const [data, setData] = useState<Payload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [horizon, setHorizon] = useState<Horizon>(15);
  const [updated, setUpdated] = useState<Date | null>(null);

  useEffect(() => {
    fetch("/taxi_zones.geojson")
      .then((res) => {
        if (!res.ok) throw new Error(`zones map ${res.status}`);
        return res.json();
      })
      .then(setGeo)
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const res = await fetch("/api/zones");
        if (!res.ok) throw new Error(`API ${res.status}`);
        const body = (await res.json()) as Payload;
        if (!stop) {
          setData(body);
          setError(null);
          setUpdated(new Date());
        }
      } catch (err) {
        if (!stop) setError(err instanceof Error ? err.message : "API unreachable");
      }
    };
    void tick();
    const id = window.setInterval(() => void tick(), 2000);
    return () => {
      stop = true;
      window.clearInterval(id);
    };
  }, []);

  const lookup = useMemo(() => {
    const map = new Map<number, Zone>();
    for (const zone of data?.zones ?? []) map.set(zone.zone_id, zone);
    return map;
  }, [data]);

  const ranked = useMemo(() => {
    return [...(data?.zones ?? [])]
      .filter((zone) => valueOf(zone, horizon) != null)
      .sort((a, b) => (valueOf(b, horizon) ?? 0) - (valueOf(a, horizon) ?? 0))
      .slice(0, 8);
  }, [data, horizon]);

  const max = ranked.length ? (valueOf(ranked[0], horizon) ?? 1) : 1;
  const model = horizon === 15 ? data?.model_15 : data?.model_30;
  const live = (data?.zones.length ?? 0) > 0 && error == null;
  const signature = `${horizon}:${ranked.map((zone) => fmt(valueOf(zone, horizon))).join(",")}`;

  return (
    <div className="app">
      <div className="map-wrap">
        <MapContainer center={[40.73, -73.94]} zoom={11} zoomControl={false}>
          <TileLayer
            attribution='&copy; OpenStreetMap &copy; CARTO'
            url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
          />
          {geo && (
            <GeoJSON
              key={signature}
              data={geo}
              style={(feature) => {
                const id = Number((feature as Feature)?.properties?.LocationID);
                const zone = lookup.get(id);
                const value = zone ? valueOf(zone, horizon) : null;
                return {
                  color: "#0c1016",
                  weight: 0.7,
                  fillColor: value == null ? "#1a2330" : mix(value / Math.max(max, 1)),
                  fillOpacity: value == null ? 0.35 : 0.9,
                };
              }}
              onEachFeature={(feature, layer: Layer) => {
                const props = (feature as Feature).properties ?? {};
                const id = Number(props.LocationID);
                const zone = lookup.get(id);
                const name = zone?.name || String(props.zone || `Zone ${id}`);
                const value = zone ? valueOf(zone, horizon) : null;
                const recent = zone ? `${fmt(zone.pu_15m)} recent 15m` : "no features yet";
                layer.bindTooltip(
                  `<strong>${name}</strong><br/>${value == null ? "forecast unavailable" : `${fmt(value)} predicted`}<br/>${recent}`,
                );
              }}
            />
          )}
        </MapContainer>
      </div>
      <aside className="panel">
        <header>
          <p className="kicker">Yellow taxi · TLC January 2024</p>
          <h1>Surgecast</h1>
          <p className="sub">Predicted pickups by taxi zone, read from the live feature store.</p>
        </header>
        <div className="status-row">
          <span className={error ? "pill bad" : live ? "pill live" : "pill"}>{error ? "api down" : live ? "live" : "waiting"}</span>
          <span className="pill">{updated ? updated.toLocaleTimeString() : "—"}</span>
        </div>
        {error && <div className="banner">{error}. The map keeps the last successful reading.</div>}
        {!error && data && data.zones.length === 0 && (
          <div className="banner">Replay has not closed a 5-minute bucket yet. Zones appear after the first watermark advance.</div>
        )}
        <div className="stats">
          <div className="stat">
            <span>Event time</span>
            <strong>{data?.event_time ? data.event_time.replace("T", " ").slice(0, 16) : "—"}</strong>
          </div>
          <div className="stat">
            <span>Model</span>
            <strong>{model == null ? "not loaded" : `v${model} · ${horizon}m`}</strong>
          </div>
          <div className="stat">
            <span>Zones with features</span>
            <strong>{data?.zones.length ?? 0}</strong>
          </div>
          <div className="stat">
            <span>Peak forecast</span>
            <strong>{ranked.length ? fmt(valueOf(ranked[0], horizon)) : "—"}</strong>
          </div>
        </div>
        <div className="toggle">
          <button className={horizon === 15 ? "on" : ""} onClick={() => setHorizon(15)} type="button">
            Next 15 min
          </button>
          <button className={horizon === 30 ? "on" : ""} onClick={() => setHorizon(30)} type="button">
            Next 30 min
          </button>
        </div>
        <div className="legend">
          <div className="bar" />
          <p>
            <span>quiet</span>
            <span>hot</span>
          </p>
        </div>
        <div className="list">
          <h2>Hot zones</h2>
          {ranked.length === 0 && <p className="sub">No forecasts yet.</p>}
          {ranked.map((zone, index) => (
            <div className="zone" key={zone.zone_id}>
              <span className="rank">{String(index + 1).padStart(2, "0")}</span>
              <div>
                <b>{zone.name}</b>
                <em>{zone.borough}</em>
              </div>
              <div className="nums">
                {fmt(valueOf(zone, horizon))}
                <small>{fmt(zone.pu_15m)} last 15m</small>
              </div>
            </div>
          ))}
        </div>
        <div className="foot">
          <p>Pickups, not fares. Replay clock is the trip timestamp, not wall time.</p>
        </div>
      </aside>
    </div>
  );
}
