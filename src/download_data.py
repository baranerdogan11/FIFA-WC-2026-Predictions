"""Download all raw data sources into data/raw/.

Sources:
- martj42/international_results (GitHub): results.csv, shootouts.csv
- eloratings.net: current World Football Elo snapshot
- api.fifa.com: official WC2026 match calendar (competition 17, season 285023)
"""
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "results.csv": "https://raw.githubusercontent.com/martj42/international_results/master/results.csv",
    "shootouts.csv": "https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv",
    "elo_world.tsv": "http://www.eloratings.net/World.tsv",
    "fifa_matches.json": (
        "https://api.fifa.com/api/v3/calendar/matches"
        "?idCompetition=17&idSeason=285023&count=150&language=en"
    ),
}


def main() -> None:
    for name, url in SOURCES.items():
        dest = RAW / name
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
        print(f"downloaded {name} ({dest.stat().st_size / 1024:.0f} KB)")
    print(f"\nall files in {RAW}")
    print("NOTE: results.csv updates daily. build_dataset.py enforces the "
          "2026-07-03 cutoff, so newer rows are ignored automatically.")


if __name__ == "__main__":
    main()
