"""Monte Carlo simulation of the WC2026 knockout bracket from the Round of 16.

All 8 R16 ties are simulated, including Morocco-Canada and France-Paraguay
(held out per project setup). Match probabilities come from the trained
XGBoost model using team state frozen at the 2026-07-03 cutoff. Neutral-venue
predictions are symmetrized by averaging both team orderings. Knockout draws
go to penalties with a mild Elo-based tilt.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
N_SIMS = 20_000
RNG = np.random.default_rng(42)

# Sofascore team names -> results.csv names
XG_NAME_MAP = {"USA": "United States"}

# (slot, teamA, teamB, venue_country, date)
R16 = [
    (89, "Paraguay", "France", "USA", "2026-07-04"),
    (90, "Canada", "Morocco", "USA", "2026-07-04"),
    (91, "Brazil", "Norway", "USA", "2026-07-05"),
    (92, "Mexico", "England", "MEX", "2026-07-06"),
    (93, "Portugal", "Spain", "USA", "2026-07-06"),
    (94, "United States", "Belgium", "USA", "2026-07-07"),
    (95, "Argentina", "Egypt", "USA", "2026-07-07"),
    (96, "Switzerland", "Colombia", "CAN", "2026-07-07"),
]
QF = [(97, 89, 90, "2026-07-09"), (98, 93, 94, "2026-07-10"),
      (99, 91, 92, "2026-07-11"), (100, 95, 96, "2026-07-12")]
SF = [(101, 97, 98, "2026-07-14"), (102, 99, 100, "2026-07-15")]
FINAL = (104, 101, 102, "2026-07-19")

HOST = {"USA": "United States", "MEX": "Mexico", "CAN": "Canada"}


def load(xg_blend: float = 0.0):
    booster = xgb.Booster()
    booster.load_model(ROOT / "models" / "xgb_wc2026.json")
    features = json.load(open(ROOT / "models" / "features.json"))
    ts = json.load(open(PROC / "team_state.json"))
    state = ts["state"]
    if xg_blend > 0:
        # blend goals-based recent form with cutoff-safe tournament xG rates:
        # xG estimates the same latent attack/defence rates with less
        # finishing noise (Sofascore per-match data, holdout games excluded)
        tstats = pd.read_csv(PROC / "tournament_team_stats.csv")
        for r in tstats.itertuples():
            name = XG_NAME_MAP.get(r.team, r.team)
            s = state[name]
            s["gf5"] = (1 - xg_blend) * s["gf5"] + xg_blend * r.xg_for
            s["ga5"] = (1 - xg_blend) * s["ga5"] + xg_blend * r.xg_against
    return booster, features, state, ts["h2h"]


def match_probs(booster, features, state, h2h, home, away, venue, date):
    """P(home win, draw, away win) in 90'. Symmetrized when neutral."""
    def vec(h, a, neutral):
        sh, sa = state[h], state[a]
        key = "|".join(sorted((h, a)))
        hh = h2h.get(key, [])
        hw, aw = hh.count(h), hh.count(a)
        elo_h, elo_a = sh["elo"], sa["elo"]
        return {
            "elo_home": elo_h, "elo_away": elo_a,
            "elo_diff": (elo_h + (0 if neutral else 100.0)) - elo_a,
            "neutral": int(neutral), "tourn_cat": 3,
            "form5_ppg_h": sh["form5_ppg"], "form5_ppg_a": sa["form5_ppg"],
            "form10_ppg_h": sh["form10_ppg"], "form10_ppg_a": sa["form10_ppg"],
            "gf5_h": sh["gf5"], "ga5_h": sh["ga5"],
            "gf5_a": sa["gf5"], "ga5_a": sa["ga5"],
            "elo_mom_h": sh["elo_mom"], "elo_mom_a": sa["elo_mom"],
            "rest_h": (pd.Timestamp(date) - pd.Timestamp(sh["last_played"])).days,
            "rest_a": (pd.Timestamp(date) - pd.Timestamp(sa["last_played"])).days,
            "h2h_n": len(hh),
            "h2h_home_edge": (hw - aw) / len(hh) if hh else 0.0,
        }

    host = HOST.get(venue)
    if host == home:
        rows = [vec(home, away, neutral=False)]
        flip = [False]
    elif host == away:
        rows = [vec(away, home, neutral=False)]
        flip = [True]
    else:  # neutral: average both orderings
        rows = [vec(home, away, True), vec(away, home, True)]
        flip = [False, True]

    X = pd.DataFrame(rows)[features]
    P = booster.predict(xgb.DMatrix(X))
    out = np.zeros(3)
    for p, fl in zip(P, flip):
        out += p[::-1] if fl else p
    return out / len(rows)


def pens_p_home(state, home, away):
    d = state[home]["elo"] - state[away]["elo"]
    return 1.0 / (1.0 + 10 ** (-d / 1000.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xg", type=float, default=0.0, metavar="W",
                    help="blend weight for tournament xG in form features "
                         "(0=goals only, 0.5=equal blend)")
    args = ap.parse_args()
    booster, features, state, h2h = load(xg_blend=args.xg)
    if args.xg > 0:
        print(f"[xG blend active: weight={args.xg}]")

    # precompute probabilities for every possible pairing per slot
    slot_teams = {s: [a, b] for s, a, b, _, _ in R16}
    probs = {}
    for s, a, b, venue, date in R16:
        probs[(s, a, b)] = match_probs(booster, features, state, h2h, a, b, venue, date)
    rounds = [(QF, slot_teams), ]
    # build possible team sets per slot progressively
    poss = dict(slot_teams)
    for rnd in (QF, SF, [FINAL]):
        for s, sa, sb, date in rnd:
            poss[s] = poss[sa] + poss[sb]
            for a, b in itertools.product(poss[sa], poss[sb]):
                probs[(s, a, b)] = match_probs(booster, features, state, h2h, a, b, "USA", date)

    pens = {(a, b): pens_p_home(state, a, b)
            for a in state for b in state
            if a in poss[104] and b in poss[104] and a != b}

    teams = poss[104]
    counters = {r: {t: 0 for t in teams} for r in ["QF", "SF", "F", "W"]}
    r16_win = {t: 0 for t in teams}

    for _ in range(N_SIMS):
        winners = {}
        for s, a, b, venue, date in R16:
            w = play(probs[(s, a, b)], a, b, pens)
            winners[s] = w
            r16_win[w] += 1
            counters["QF"][w] += 1
        for rnd, tag in ((QF, "SF"), (SF, "F")):
            for s, sa, sb, date in rnd:
                a, b = winners[sa], winners[sb]
                w = play(probs[(s, a, b)], a, b, pens)
                winners[s] = w
                counters[tag][w] += 1
        s, sa, sb, date = FINAL
        a, b = winners[sa], winners[sb]
        w = play(probs[(s, a, b)], a, b, pens)
        counters["W"][w] += 1

    print(f"=== {N_SIMS:,} bracket simulations, team state frozen at 2026-07-03 ===\n")
    print("R16 advance probabilities:")
    for s, a, b, venue, date in R16:
        pa = r16_win[a] / N_SIMS
        p = probs[(s, a, b)]
        print(f"  {a:>14} {pa:5.1%} vs {1-pa:5.1%} {b:<14}  "
              f"(90': {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f})")

    print(f"\n{'team':<15} {'reach SF':>9} {'reach F':>9} {'CHAMPION':>10}")
    for t in sorted(teams, key=lambda t: -counters['W'][t]):
        print(f"{t:<15} {counters['SF'][t]/N_SIMS:9.1%} "
              f"{counters['F'][t]/N_SIMS:9.1%} {counters['W'][t]/N_SIMS:10.1%}")

    out = pd.DataFrame({
        "team": teams,
        "p_QF": [counters["QF"][t] / N_SIMS for t in teams],
        "p_SF": [counters["SF"][t] / N_SIMS for t in teams],
        "p_final": [counters["F"][t] / N_SIMS for t in teams],
        "p_champion": [counters["W"][t] / N_SIMS for t in teams],
    }).sort_values("p_champion", ascending=False)
    (ROOT / "outputs").mkdir(exist_ok=True)
    fname = "championship_probabilities_xg.csv" if args.xg > 0 else "championship_probabilities.csv"
    out.to_csv(ROOT / "outputs" / fname, index=False)
    print(f"\nsaved -> outputs/{fname}")


def play(p, a, b, pens):
    u = RNG.random()
    if u < p[0]:
        return a
    if u < p[0] + p[2]:
        return b
    return a if RNG.random() < pens[(a, b)] else b


if __name__ == "__main__":
    main()
