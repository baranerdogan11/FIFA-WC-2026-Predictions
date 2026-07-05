"""Train XGBoost match-outcome model (home win / draw / away win).

Validation is strictly temporal: train <2023, validate 2023-2025, then
refit on all data through the 2026-07-03 cutoff at the best iteration.
Baselines: Elo-only logistic regression and a constant-prior model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUTM = ROOT / "models"
OUTM.mkdir(exist_ok=True)

FEATURES = [
    "elo_home", "elo_away", "elo_diff", "neutral", "tourn_cat",
    "form5_ppg_h", "form5_ppg_a", "form10_ppg_h", "form10_ppg_a",
    "gf5_h", "ga5_h", "gf5_a", "ga5_a",
    "elo_mom_h", "elo_mom_a", "rest_h", "rest_a",
    "h2h_n", "h2h_home_edge",
]

PARAMS = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "reg_lambda": 2.0,
    "seed": 42,
}


def time_decay_weights(dates: pd.Series, ref: pd.Timestamp, half_life_years: float = 10.0) -> np.ndarray:
    age = (ref - dates).dt.days / 365.25
    return np.power(0.5, age / half_life_years).values


def main() -> None:
    df = pd.read_parquet(PROC / "match_features.parquet")
    cutoff = df["date"].max()

    train = df[df["date"] < "2023-01-01"]
    valid = df[(df["date"] >= "2023-01-01") & (df["date"] < "2026-01-01")]
    test26 = df[df["date"] >= "2026-01-01"]
    print(f"train={len(train)}  valid={len(valid)}  test2026={len(test26)}")

    dtrain = xgb.DMatrix(train[FEATURES], label=train["target"],
                         weight=time_decay_weights(train["date"], cutoff))
    dvalid = xgb.DMatrix(valid[FEATURES], label=valid["target"])
    dtest = xgb.DMatrix(test26[FEATURES], label=test26["target"])

    booster = xgb.train(PARAMS, dtrain, num_boost_round=3000,
                        evals=[(dvalid, "valid")],
                        early_stopping_rounds=100, verbose_eval=False)
    best_iter = booster.best_iteration
    print(f"best iteration: {best_iter}")

    def report(name, mat, y):
        p = booster.predict(mat, iteration_range=(0, best_iter + 1))
        print(f"  {name}: logloss={log_loss(y, p, labels=[0,1,2]):.4f} "
              f"acc={accuracy_score(y, p.argmax(1)):.3f}")
        return p

    print("XGBoost:")
    report("valid 2023-25", dvalid, valid["target"])
    report("test  2026   ", dtest, test26["target"])

    # baselines on the same splits
    prior = train["target"].value_counts(normalize=True).sort_index().values
    print("Constant-prior baseline:")
    for name, part in [("valid", valid), ("test26", test26)]:
        p = np.tile(prior, (len(part), 1))
        print(f"  {name}: logloss={log_loss(part['target'], p, labels=[0,1,2]):.4f}")
    lr = LogisticRegression(max_iter=1000, multi_class="multinomial")
    lr.fit(train[["elo_diff"]], train["target"])
    print("Elo-only logistic baseline:")
    for name, part in [("valid", valid), ("test26", test26)]:
        p = lr.predict_proba(part[["elo_diff"]])
        print(f"  {name}: logloss={log_loss(part['target'], p, labels=[0,1,2]):.4f} "
              f"acc={accuracy_score(part['target'], p.argmax(1)):.3f}")

    # refit on ALL data through cutoff at best_iter
    dall = xgb.DMatrix(df[FEATURES], label=df["target"],
                       weight=time_decay_weights(df["date"], cutoff))
    final = xgb.train(PARAMS, dall, num_boost_round=best_iter + 1)
    final.save_model(OUTM / "xgb_wc2026.json")
    with open(OUTM / "features.json", "w") as f:
        json.dump(FEATURES, f)
    print(f"\nfinal model refit on {len(df)} matches -> models/xgb_wc2026.json")

    imp = final.get_score(importance_type="gain")
    print("\ntop feature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:8]:
        print(f"  {k:<16} {v:8.1f}")


if __name__ == "__main__":
    main()
