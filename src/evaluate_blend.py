"""Backtest: does historical data improve accuracy on the 2026 tournament?

- XGBoost trained ONLY on matches before the tournament (< 2026-06-11),
  so every tournament match is out-of-sample for it.
- Stats model rebuilt per match from an expanding window of prior
  tournament matches only (no future information).
- Both evaluated on tournament matches from matchday 2 onward (both teams
  need at least one prior match for stats ratings), through the cutoff.
- Blends P = (1-w)*P_xgb + w*P_stats scored across w to pick the weight.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from stats_model import NAME_MAP, SCORE_FIX, _load_matches, build_ratings, match_probs_stats

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

FEATURES = [
    "elo_home", "elo_away", "elo_diff", "neutral", "tourn_cat",
    "form5_ppg_h", "form5_ppg_a", "form10_ppg_h", "form10_ppg_a",
    "gf5_h", "ga5_h", "gf5_a", "ga5_a",
    "elo_mom_h", "elo_mom_a", "rest_h", "rest_a",
    "h2h_n", "h2h_home_edge",
]
PARAMS = {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss",
          "max_depth": 4, "learning_rate": 0.03, "subsample": 0.8,
          "colsample_bytree": 0.8, "min_child_weight": 20, "reg_lambda": 2.0,
          "seed": 42}


def main() -> None:
    feat = pd.read_parquet(PROC / "match_features.parquet")
    train = feat[feat["date"] < "2026-06-11"]
    wc = feat[(feat["date"] >= "2026-06-11") & (feat["tournament"] == "FIFA World Cup")].copy()

    # train XGB strictly pre-tournament
    ref = train["date"].max()
    age = (ref - train["date"]).dt.days / 365.25
    w = np.power(0.5, age / 10.0)
    dtrain = xgb.DMatrix(train[FEATURES], label=train["target"], weight=w)
    booster = xgb.train(PARAMS, dtrain, num_boost_round=350)

    # sofascore matches with results.csv naming
    sofa = _load_matches()
    sofa["home"] = sofa["home"].map(lambda t: NAME_MAP.get(t, t))
    sofa["away"] = sofa["away"].map(lambda t: NAME_MAP.get(t, t))
    sofa = sofa.sort_values("date_utc").reset_index(drop=True)
    sofa["key"] = sofa.apply(lambda r: frozenset((r.home, r.away)), axis=1)

    # evaluation set: tournament matches where both teams have >=1 prior match
    y, p_xgb, p_sts = [], [], []
    skipped = 0
    for m in wc.sort_values("date").itertuples():
        key = frozenset((m.home_team, m.away_team))
        idx = sofa.index[sofa["key"] == key]
        if len(idx) == 0:
            skipped += 1
            continue
        prior = sofa.iloc[: idx[0]]
        teams_seen = set(prior.home) | set(prior.away)
        if m.home_team not in teams_seen or m.away_team not in teams_seen:
            skipped += 1
            continue
        ratings = build_ratings(prior)
        p_sts.append(match_probs_stats(ratings, m.home_team, m.away_team))
        row = pd.DataFrame([{f: getattr(m, f) for f in FEATURES}])
        p_xgb.append(booster.predict(xgb.DMatrix(row[FEATURES]))[0])
        y.append(m.target)

    y = np.array(y)
    p_xgb = np.array(p_xgb)
    p_sts = np.array(p_sts)
    print(f"backtest matches: {len(y)} (skipped {skipped}: matchday 1 / no priors)\n")
    print(f"{'blend w (stats share)':>22} {'log loss':>9} {'accuracy':>9}")
    best = (None, 9e9)
    for wgt in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        p = (1 - wgt) * p_xgb + wgt * p_sts
        ll = log_loss(y, p, labels=[0, 1, 2])
        acc = (p.argmax(1) == y).mean()
        tag = "  <- XGB only" if wgt == 0 else ("  <- stats only" if wgt == 1 else "")
        print(f"{wgt:>22.1f} {ll:>9.4f} {acc:>9.3f}{tag}")
        if ll < best[1]:
            best = (wgt, ll)
    print(f"\nbest blend: w={best[0]} (log loss {best[1]:.4f})")


if __name__ == "__main__":
    main()
