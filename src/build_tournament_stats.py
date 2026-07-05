"""Aggregate per-match Sofascore stats (xG, possession, shots) into
cutoff-safe team-level tournament stats for the 16 R16 teams.

Excludes the two held-out R16 games (Canada-Morocco, Paraguay-France,
played 2026-07-04). Sofascore dates are UTC, so exclusion is by fixture,
not by date (Colombia-Ghana shows 07-04 UTC but was played 07-03 local).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = [{"Canada", "Morocco"}, {"Paraguay", "France"}]

R16 = ["Argentina", "Belgium", "Brazil", "Canada", "Colombia", "Egypt",
       "England", "France", "Mexico", "Morocco", "Norway", "Paraguay",
       "Portugal", "Spain", "Switzerland", "USA"]


def main() -> None:
    df = pd.read_csv(ROOT / "data" / "raw" / "sofascore_match_stats.csv")
    df = df[~df.apply(lambda r: {r.home, r.away} in HELD_OUT, axis=1)]
    print(f"matches after holdout exclusion: {len(df)}")

    rows = []
    for team in R16:
        h = df[df.home == team]
        a = df[df.away == team]
        n = len(h) + len(a)
        rows.append({
            "team": team, "matches": n,
            "xg_for": round((h.xg_h.sum() + a.xg_a.sum()) / n, 3),
            "xg_against": round((h.xg_a.sum() + a.xg_h.sum()) / n, 3),
            "goals_for": (h.home_score.sum() + a.away_score.sum()) / n,
            "goals_against": (h.away_score.sum() + a.home_score.sum()) / n,
            "shots_for": round((h.shots_h.sum() + a.shots_a.sum()) / n, 1),
            "shots_against": round((h.shots_a.sum() + a.shots_h.sum()) / n, 1),
            "sot_for": round((h.sot_h.sum() + a.sot_a.sum()) / n, 1),
            "possession": round((h.poss_h.sum() + a.poss_a.sum()) / n, 1),
        })
    out = pd.DataFrame(rows).sort_values("xg_for", ascending=False)
    out["xg_diff"] = (out.xg_for - out.xg_against).round(3)
    dest = ROOT / "data" / "processed" / "tournament_team_stats.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved -> {dest}")

    # note: shootout matches store aggregate scores incl. pens on Sofascore
    # (Germany-Paraguay 4-5, Netherlands-Morocco 3-4, Australia-Egypt 3-5);
    # goals_for/against here are slightly inflated for those teams and are
    # informational only -- the model's goal features come from results.csv.


if __name__ == "__main__":
    main()
