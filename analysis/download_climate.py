"""
Downloads the hourly climate records used in the paper into data/climate/.

Two independent sources, so results can be cross-checked:
  1. Meteostat bulk hourly data (weather-station observations)
       https://data.meteostat.net/hourly/{year}/{station}.csv.gz
  2. NASA POWER hourly data (gridded reanalysis at the same coordinates)
       https://power.larc.nasa.gov/api/temporal/hourly/point

Uses only the Python standard library. Run from the project folder:

    python analysis/download_climate.py

Files already downloaded are skipped, so it is safe to run again.
A log of every request is written to data/climate/download_log.csv.
"""
import csv
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

YEARS = range(2015, 2026)

# id, name, latitude, longitude, elevation (m).
# Taken from the Meteostat station directory
# (github.com/meteostat/weather-stations), country code GT.
STATIONS = [
    ("78641", "Guatemala La Aurora", 14.5833, -90.5167, 1489),
    ("78640", "Guatemala Observatorio Nacional", 14.5833, -90.5167, 1502),
    ("MGQZ0", "Quetzaltenango", 14.8333, -91.5167, 2500),
    ("78627", "Huehuetenango", 15.3167, -91.4667, 1901),
    ("78631", "Coban", 15.4667, -90.3167, 1316),
    ("78644", "Zacapa", 14.9667, -89.5333, 490),
    ("78639", "Retalhuleu", 14.5333, -91.6667, 239),
    ("MGCP0", "Champerico", 14.3069, -91.8994, 8),
    ("78647", "San Jose", 13.9167, -90.8167, 2),
    ("78637", "Puerto Barrios", 15.7167, -88.6000, 1),
    ("78615", "Flores", 16.9167, -89.8833, 115),
    ("MGMM0", "Mundo Maya (Flores)", 16.9139, -89.8664, 122),
    ("MGPC0", "Paso Caballos", 17.2603, -90.2639, 50),
]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "climate"
HEADERS = {"User-Agent": "paper2-stability-research/1.0 (academic use)"}


def fetch(url: str, dest: Path, log: list) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.append((url, dest.name, "skipped (already present)", dest.stat().st_size))
        return
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        if not data:
            raise ValueError("empty response")
        dest.write_bytes(data)
        log.append((url, dest.name, "ok", len(data)))
        print(f"  ok       {dest.name}  ({len(data) / 1024:.0f} KB)")
    except urllib.error.HTTPError as err:
        log.append((url, dest.name, f"HTTP {err.code}", 0))
        print(f"  no data  {dest.name}  (HTTP {err.code})")
    except Exception as err:  # network down, timeout, etc.
        log.append((url, dest.name, f"error: {err}", 0))
        print(f"  ERROR    {dest.name}  ({err})")
    time.sleep(0.4)


def main() -> None:
    (OUT / "meteostat").mkdir(parents=True, exist_ok=True)
    (OUT / "nasa_power").mkdir(parents=True, exist_ok=True)
    log: list = []

    with open(OUT / "stations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["station_id", "name", "latitude", "longitude", "elevation_m"])
        w.writerows(STATIONS)

    print("Meteostat (station observations)")
    for sid, name, *_ in STATIONS:
        for year in YEARS:
            fetch(f"https://data.meteostat.net/hourly/{year}/{sid}.csv.gz",
                  OUT / "meteostat" / f"{sid}_{year}.csv.gz", log)

    print("NASA POWER (gridded reanalysis)")
    done = set()
    for sid, name, lat, lon, _ in STATIONS:
        if (lat, lon) in done:
            continue
        done.add((lat, lon))
        for year in YEARS:
            url = ("https://power.larc.nasa.gov/api/temporal/hourly/point"
                   f"?parameters=T2M,RH2M&community=RE&longitude={lon}&latitude={lat}"
                   f"&start={year}0101&end={year}1231&format=CSV&time-standard=UTC")
            fetch(url, OUT / "nasa_power" / f"{sid}_{year}.csv", log)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT / "download_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["downloaded_utc", "url", "file", "status", "bytes"])
        w.writerows([(stamp, *row) for row in log])

    ok = sum(1 for r in log if r[2] == "ok" or r[2].startswith("skipped"))
    print(f"\n{ok} of {len(log)} files available in {OUT}")
    print("Log written to data/climate/download_log.csv")


if __name__ == "__main__":
    main()
