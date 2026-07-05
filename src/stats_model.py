"""Tournament stats model: match probabilities from xG, shots, shots on
target, and possession only.

Built from the 88 pre-cutoff WC2026 matches (Sofascore per-match data,
holdout games excluded). For every team:

1. A composite per-match scoring estimate blends the three chance metrics:
   0.55*xG + 0.30*SoT*conv_sot + 0.15*shots*conv_shot, where the conversion
   rates are tournament-wide goals-per-SoT and goals-per-shot.
2. Possession enters as a multiplicative control factor (poss/50)^0.25.
3. Attack/defence ratings are opponent-adjusted iteratively (a strong xG
   number against Spain counts for more than one against Qatar).
4. A hypothetical matchup A vs B becomes Poisson rates
   lambda_A = mu * att_A * def_B, giving P(win/draw/loss) over a score grid.

Scores for the three shootout matches are corrected to their true 120'
results (Sofascore stores shootout-inflated aggregates).
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HELD_OUT = [{"Canada", "Morocco"}, {"Paraguay", "France"}]
SCORE_FIX = {  # fixtures decided on penalties: true score after 120'
    frozenset(("Germany", "Paraguay")): (1, 1),
    frozenset(("Netherlands", "Morocco")): (1, 1),
    frozenset(("Australia", "Egypt")): (1, 1),
}
NAME_MAP = {"USA": "United States"}  # Sofascore -> results.csv names

W_XG, W_SOT, W_SH = 0.55, 0.30, 0.15
POSS_EXP = 0.25
MAX_GOALS = 9


def _load_matches() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "raw" / "sofascore_match_stats.csv")
    df = df[~df.apply(lambda r: {r.home, r.away} in HELD_OUT, axis=1)].copy()
    for key, (hs, as_) in SCORE_FIX.items():
        m = df.apply(lambda r: frozenset((r.home, r.away)) == key, axis=1)
        df.loc[m, ["home_score", "away_score"]] = (hs, as_)
    return df


def build_ratings() -> dict:
    df = _load_matches()
    goals = df.home_score.sum() + df.away_score.sum()
    conv_sot = goals / (df.sot_h.sum() + df.sot_a.sum())
    conv_shot = goals / (df.shots_h.sum() + df.shots_a.sum())
    mu = goals / (2 * len(df))  # avg goals per team per match

    # per-team match lists of (composite_for, composite_against, opponent)
    rows = []
    for r in df.itertuples():
        ch = W_XG * r.xg_h + W_SOT * r.sot_h * conv_sot + W_SH * r.shots_h * conv_shot
        ca = W_XG * r.xg_a + W_SOT * r.sot_a * conv_sot + W_SH * r.shots_a * conv_shot
        ch *= (r.poss_h / 50.0) ** POSS_EXP
        ca *= (r.poss_a / 50.0) ** POSS_EXP
        rows.append((r.home, r.away, ch, ca))
        rows.append((r.away, r.home, ca, ch))

    teams = sorted({t for t, *_ in rows})
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}
    for _ in range(8):  # iterative opponent adjustment
        new_att, new_dfn = {}, {}
        for t in teams:
            ms = [(o, cf, ca) for tt, o, cf, ca in rows if tt == t]
            new_att[t] = np.mean([cf / (mu * dfn[o]) for o, cf, _ in ms])
            new_dfn[t] = np.mean([ca / (mu * att[o]) for o, _, ca in ms])
        m_a = np.mean(list(new_att.values()))
        m_d = np.mean(list(new_dfn.values()))
        att = {t: v / m_a for t, v in new_att.items()}
        dfn = {t: v / m_d for t, v in new_dfn.items()}

    out = {NAME_MAP.get(t, t): {"att": att[t], "def": dfn[t]} for t in teams}
    out["_mu"] = mu
    return out


def match_probs_stats(ratings: dict, team_a: str, team_b: str) -> np.ndarray:
    """P(A wins, draw, B wins) in 90' from Poisson score grid."""
    mu = ratings["_mu"]
    la = mu * ratings[team_a]["att"] * ratings[team_b]["def"]
    lb = mu * ratings[team_b]["att"] * ratings[team_a]["def"]
    k = np.arange(MAX_GOALS + 1)
    fact = np.array([math.factorial(i) for i in k], dtype=float)
    pa = np.exp(-la) * la ** k / fact
    pb = np.exp(-lb) * lb ** k / fact
    grid = np.outer(pa, pb)
    grid /= grid.sum()
    return np.array([np.tril(grid, -1).sum(),   # A scores more
                     np.trace(grid),            # draw
                     np.triu(grid, 1).sum()])   # B scores more


if __name__ == "__main__":
    r = build_ratings()
    print(f"mu = {r['_mu']:.3f} goals/team/match\n")
    r16 = ["Morocco", "Canada", "France", "Paraguay", "Brazil", "Norway",
           "Mexico", "England", "Portugal", "Spain", "United States",
           "Belgium", "Argentina", "Egypt", "Switzerland", "Colombia"]
    print(f"{'team':<15} {'attack':>7} {'defence':>8}  (defence <1 = good)")
    for t in sorted(r16, key=lambda t: -(r[t]["att"] / r[t]["def"])):
        print(f"{t:<15} {r[t]['att']:7.2f} {r[t]['def']:8.2f}")
    print("\nheld-out ties (stats model only, 90'):")
    for a, b in [("Morocco", "Canada"), ("France", "Paraguay")]:
        p = match_probs_stats(r, a, b)
        print(f"  {a} v {b}: {p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}")
