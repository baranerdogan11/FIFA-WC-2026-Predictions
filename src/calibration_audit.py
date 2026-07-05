"""Calibration audit of the match-outcome model on out-of-time 2026 data.

Trains on matches < 2026 (same spec as train_model.py), audits the 399
pre-cutoff 2026 matches:
  1. Per-class reliability (predicted probability vs observed frequency)
  2. Expected calibration error (ECE) per class
  3. Draw-rate check (mean predicted vs actual)
  4. Temperature scaling fitted on 2023-25 validation; kept only if it
     improves 2026 log loss.
Saves a reliability diagram to outputs/calibration.png.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import log_loss

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
CLASSES = ["home win", "draw", "away win"]
N_BINS = 8


def ece(p: np.ndarray, hit: np.ndarray, n_bins: int = N_BINS):
    """quantile-binned expected calibration error + bin table"""
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    qs[0], qs[-1] = 0, 1
    rows, err = [], 0.0
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        conf, obs = p[m].mean(), hit[m].mean()
        rows.append((conf, obs, int(m.sum())))
        err += m.mean() * abs(conf - obs)
    return err, rows


def main() -> None:
    df = pd.read_parquet(PROC / "match_features.parquet")
    train = df[df["date"] < "2026-01-01"]
    valid = df[(df["date"] >= "2023-01-01") & (df["date"] < "2026-01-01")]
    test = df[df["date"] >= "2026-01-01"]
    print(f"train<2026: {len(train)}  |  audit set 2026: {len(test)}")

    ref = train["date"].max()
    w = np.power(0.5, ((ref - train["date"]).dt.days / 365.25) / 10.0)
    booster = xgb.train(PARAMS, xgb.DMatrix(train[FEATURES], label=train["target"], weight=w),
                        num_boost_round=350)

    def margins(part):
        return booster.predict(xgb.DMatrix(part[FEATURES]), output_margin=True)

    y_te = test["target"].values
    m_te = margins(test)
    p_te = softmax(m_te, axis=1)

    print(f"\n2026 log loss (uncalibrated): {log_loss(y_te, p_te, labels=[0,1,2]):.4f}")
    print(f"\n{'class':<10} {'mean pred':>10} {'observed':>10} {'ECE':>7}")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for c in range(3):
        p_c, hit_c = p_te[:, c], (y_te == c).astype(float)
        e, rows = ece(p_c, hit_c)
        print(f"{CLASSES[c]:<10} {p_c.mean():>10.3f} {hit_c.mean():>10.3f} {e:>7.3f}")
        ax = axes[c]
        confs, obss, ns = zip(*rows)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.plot(confs, obss, "o-", color="#1f6fb2")
        for x, yy, n in rows:
            ax.annotate(str(n), (x, yy), fontsize=7, xytext=(3, 4),
                        textcoords="offset points")
        ax.set_title(f"{CLASSES[c]} (ECE {e:.3f})")
        ax.set_xlabel("predicted probability")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Reliability -- 2026 out-of-time matches (n=%d), bins annotated with counts" % len(test))
    fig.tight_layout()
    out = ROOT / "outputs" / "calibration.png"
    fig.savefig(out, dpi=140)
    print(f"\nreliability diagram -> {out}")

    # temperature scaling fitted on validation margins
    y_va, m_va = valid["target"].values, margins(valid)

    def nll(T):
        return log_loss(y_va, softmax(m_va / T, axis=1), labels=[0, 1, 2])

    T = minimize_scalar(nll, bounds=(0.5, 3.0), method="bounded").x
    p_te_T = softmax(m_te / T, axis=1)
    ll0 = log_loss(y_te, p_te, labels=[0, 1, 2])
    llT = log_loss(y_te, p_te_T, labels=[0, 1, 2])
    print(f"\ntemperature scaling: T={T:.3f} (fitted on 2023-25 valid)")
    print(f"2026 log loss: {ll0:.4f} -> {llT:.4f} "
          f"({'improves' if llT < ll0 - 1e-4 else 'no material improvement'})")

    # draw check
    print(f"\ndraw check: mean predicted {p_te[:,1].mean():.3f} vs actual {(y_te==1).mean():.3f}")


if __name__ == "__main__":
    main()
