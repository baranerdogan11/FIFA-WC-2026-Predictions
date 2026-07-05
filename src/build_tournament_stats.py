"""Build team-level tournament stat datasets from per-match Sofascore data.

Outputs (data/processed/):
1. team_match_stats.csv  -- long format: one row per team per match, with the
   team's own metrics AND the opponent's metrics (xG conceded, opponent
   shots, opponent shots on target, opponent possession) for every game.
   Covers all 48 teams, all 90 played matches, holdout games flagged.
2. tournament_team_stats.csv -- cutoff-safe per-team aggregates for the 16
   R16 teams, for- and against- versions of every metric. Excludes the two
   held-out R16 games (Canada-Morocco, Paraguay-France, 2026-07-04).

Sofascore dates are UTC, so holdout exclusion is by fixture, not date.
Scores for the three shootout matches are corrected to their true 120'
results (Sofascore stores shootout-inflated aggregates).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = [{"Canada", "Morocco"}, {"Paraguay", "France"}]
SCORE_FIX = {
    frozenset(("Germany", "Paraguay")): (1, 1),
    frozenset(("Netherlands", "Morocco")): (1, 1),
    frozenset(("Australia", "Egypt")): (1, 1),
}

R16 = ["Argentina", "Belgium", "Brazil", "Canada", "Colombia", "Egypt",
       "England", "France", "Mexico", "Morocco", "Norway", "Paraguay",
       "Portugal", "Spain", "Switzerland", "USA"]


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "sofascore_match_stats.csv")
    for key, (hs, as_) in SCORE_FIX.items():
        m = df.apply(lambda r: frozenset((r.home, r.away)) == key, axis=1)
        df.loc[m, ["home_score", "away_score"]] = (hs, as_)
    df["held_out"] = df.apply(lambda r: {r.home, r.away} in HELD_OUT, axis=1)
    return df


def build_long(df: pd.DataFrame) -> pd.DataFrame:
    """one row per team per match, own + opponent metrics side by side"""
    sides = []
    for own, opp in (("h", "a"), ("a", "h")):
        s = pd.DataFrame({
            "date_utc": df.date_utc,
            "team": df["home" if own == "h" else "away"],
            "opponent": df["away" if own == "h" else "home"],
            "is_home_listed": own == "h",
            "held_out": df.held_out,
            "goals_for": df["home_score" if own == "h" else "away_score"],
            "goals_against": df["away_score" if own == "h" else "home_score"],
            "xg_for": df[f"xg_{own}"],
            "xg_against": df[f"xg_{opp}"],          # xG given to the opponent
            "shots_for": df[f"shots_{own}"],
            "shots_against": df[f"shots_{opp}"],    # opponent shots
            "sot_for": df[f"sot_{own}"],
            "sot_against": df[f"sot_{opp}"],        # opponent shots on target
            "poss_for": df[f"poss_{own}"],
            "poss_against": df[f"poss_{opp}"],      # opponent possession
        })
        sides.append(s)
    long_df = pd.concat(sides).sort_values(["date_utc", "team"]).reset_index(drop=True)
    long_df["xg_diff"] = (long_df.xg_for - long_df.xg_against).round(3)
    return long_df


def main() -> None:
    df = load_matches()
    long_df = build_long(df)
    dest_long = ROOT / "data" / "processed" / "team_match_stats.csv"
    long_df.to_csv(dest_long, index=False)
    print(f"team-match rows: {len(long_df)} ({long_df.team.nunique()} teams, "
          f"{len(df)} matches, {int(long_df.held_out.sum())} holdout rows flagged)")
    print(f"saved -> {dest_long}\n")

    # cutoff-safe aggregates for the 16 R16 teams
    ok = long_df[~long_df.held_out]
    rows = []
    for team in R16:
        t = ok[ok.team == team]
        n = len(t)
        rows.append({
            "team": team, "matches": n,
            "xg_for": round(t.xg_for.mean(), 3),
            "xg_against": round(t.xg_against.mean(), 3),
            "goals_for": round(t.goals_for.mean(), 2),
            "goals_against": round(t.goals_against.mean(), 2),
            "shots_for": round(t.shots_for.mean(), 1),
            "shots_against": round(t.shots_against.mean(), 1),
            "sot_for": round(t.sot_for.mean(), 1),
            "sot_against": round(t.sot_against.mean(), 1),
            "possession": round(t.poss_for.mean(), 1),
            "possession_against": round(t.poss_against.mean(), 1),
        })
    out = pd.DataFrame(rows).sort_values("xg_for", ascending=False)
    out["xg_diff"] = (out.xg_for - out.xg_against).round(3)
    dest = ROOT / "data" / "processed" / "tournament_team_stats.csv"
    out.to_csv(dest, index=False)
    print(out.to_string(index=False))
    print(f"\nsaved -> {dest}")


if __name__ == "__main__":
    main()
