"""Build training dataset for WC2026 prediction model.

Single chronological pass over 49.5k internationals (1872-2026):
maintains per-team Elo and rolling form state, emits pre-match features
(no leakage), and snapshots team state at the cutoff for simulation.

Hard cutoff: 2026-07-03 (end of Round of 32). The two played R16 games
(Morocco-Canada, France-Paraguay, both 2026-07-04) are treated as unplayed.
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = pd.Timestamp("2026-07-03 23:59:59")
TRAIN_START = pd.Timestamp("1994-01-01")  # Elo burn-in uses everything before

HOME_ADV = 100.0  # Elo points, applied only when not neutral

K_BY_TOURNAMENT = {
    "FIFA World Cup": 60,
    "FIFA World Cup qualification": 40,
    "Copa América": 50, "African Cup of Nations": 50, "UEFA Euro": 50,
    "AFC Asian Cup": 50, "Gold Cup": 50, "CONCACAF Championship": 50,
    "Confederations Cup": 40, "UEFA Nations League": 40, "CONCACAF Nations League": 40,
    "Friendly": 20,
}
DEFAULT_K = 30


def k_factor(tournament: str) -> float:
    for key, k in K_BY_TOURNAMENT.items():
        if key.lower() in tournament.lower():
            return k
    return DEFAULT_K


def tournament_category(t: str) -> int:
    tl = t.lower()
    if tl == "friendly":
        return 0
    if "world cup qualification" in tl:
        return 2
    if tl == "fifa world cup":
        return 3
    return 1  # continental / nations league / other competitive


def margin_multiplier(gd: int) -> float:
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return 1.75 + max(0, gd - 3) / 8.0


def expected(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))


def main() -> None:
    df = pd.read_csv(RAW / "results.csv", parse_dates=["date"])
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["date"] <= CUTOFF].sort_values("date").reset_index(drop=True)
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    print(f"matches through cutoff: {len(df)}, last date: {df['date'].max().date()}")

    elo = defaultdict(lambda: 1500.0)
    recent = defaultdict(lambda: deque(maxlen=10))   # (points, gf, ga)
    elo_hist = defaultdict(lambda: deque(maxlen=10)) # elo before each match
    last_played = {}
    h2h = defaultdict(lambda: deque(maxlen=10))      # from home-alphabetical perspective

    rows = []
    for m in df.itertuples(index=False):
        h, a = m.home_team, m.away_team
        neutral = bool(m.neutral)
        eh, ea = elo[h], elo[a]

        if m.date >= TRAIN_START:
            rh, ra = recent[h], recent[a]
            key = tuple(sorted((h, a)))
            hh = h2h[key]
            h2h_h_wins = sum(1 for w in hh if w == h)
            h2h_a_wins = sum(1 for w in hh if w == a)
            rows.append({
                "date": m.date, "home_team": h, "away_team": a,
                "tournament": m.tournament,
                "elo_home": eh, "elo_away": ea,
                "elo_diff": (eh + (0 if neutral else HOME_ADV)) - ea,
                "neutral": int(neutral),
                "tourn_cat": tournament_category(m.tournament),
                "form5_ppg_h": np.mean([r[0] for r in list(rh)[-5:]]) if rh else np.nan,
                "form5_ppg_a": np.mean([r[0] for r in list(ra)[-5:]]) if ra else np.nan,
                "form10_ppg_h": np.mean([r[0] for r in rh]) if rh else np.nan,
                "form10_ppg_a": np.mean([r[0] for r in ra]) if ra else np.nan,
                "gf5_h": np.mean([r[1] for r in list(rh)[-5:]]) if rh else np.nan,
                "ga5_h": np.mean([r[2] for r in list(rh)[-5:]]) if rh else np.nan,
                "gf5_a": np.mean([r[1] for r in list(ra)[-5:]]) if ra else np.nan,
                "ga5_a": np.mean([r[2] for r in list(ra)[-5:]]) if ra else np.nan,
                "elo_mom_h": eh - elo_hist[h][0] if elo_hist[h] else 0.0,
                "elo_mom_a": ea - elo_hist[a][0] if elo_hist[a] else 0.0,
                "rest_h": min((m.date - last_played[h]).days, 60) if h in last_played else 60,
                "rest_a": min((m.date - last_played[a]).days, 60) if a in last_played else 60,
                "h2h_n": len(hh),
                "h2h_home_edge": (h2h_h_wins - h2h_a_wins) / len(hh) if hh else 0.0,
                "home_score": m.home_score, "away_score": m.away_score,
            })

        # ---- update state (post-match) ----
        gd = m.home_score - m.away_score
        res_h = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        k = k_factor(m.tournament) * margin_multiplier(gd)
        exp_h = expected(eh + (0 if neutral else HOME_ADV), ea)
        delta = k * (res_h - exp_h)
        elo_hist[h].append(eh)
        elo_hist[a].append(ea)
        elo[h] = eh + delta
        elo[a] = ea - delta
        pts_h = 3 if gd > 0 else (1 if gd == 0 else 0)
        pts_a = 3 if gd < 0 else (1 if gd == 0 else 0)
        recent[h].append((pts_h, m.home_score, m.away_score))
        recent[a].append((pts_a, m.away_score, m.home_score))
        last_played[h] = m.date
        last_played[a] = m.date
        if gd != 0:
            h2h[tuple(sorted((h, a)))].append(h if gd > 0 else a)

    feat = pd.DataFrame(rows)
    feat["target"] = np.select(
        [feat.home_score > feat.away_score, feat.home_score == feat.away_score],
        [0, 1], default=2,
    )
    feat.to_parquet(OUT / "match_features.parquet", index=False)
    print(f"feature rows: {len(feat)}  (from {feat['date'].min().date()})")

    # ---- team state snapshot at cutoff, for simulation ----
    state = {}
    for t in elo:
        r = recent[t]
        state[t] = {
            "elo": elo[t],
            "form5_ppg": float(np.mean([x[0] for x in list(r)[-5:]])) if r else np.nan,
            "form10_ppg": float(np.mean([x[0] for x in r])) if r else np.nan,
            "gf5": float(np.mean([x[1] for x in list(r)[-5:]])) if r else np.nan,
            "ga5": float(np.mean([x[2] for x in list(r)[-5:]])) if r else np.nan,
            "elo_mom": elo[t] - elo_hist[t][0] if elo_hist[t] else 0.0,
            "last_played": str(last_played[t].date()) if t in last_played else None,
        }
    h2h_out = {f"{k[0]}|{k[1]}": list(v) for k, v in h2h.items()}
    with open(OUT / "team_state.json", "w") as f:
        json.dump({"state": state, "h2h": h2h_out}, f)

    r16 = ["Morocco", "Canada", "France", "Paraguay", "Brazil", "Norway",
           "Mexico", "England", "Portugal", "Spain", "United States",
           "Belgium", "Argentina", "Egypt", "Switzerland", "Colombia"]
    print("\nElo at cutoff (R16 teams):")
    for t in sorted(r16, key=lambda x: -state[x]["elo"]):
        print(f"  {t:<15} {state[t]['elo']:7.1f}  form5_ppg={state[t]['form5_ppg']:.2f}")


if __name__ == "__main__":
    main()
