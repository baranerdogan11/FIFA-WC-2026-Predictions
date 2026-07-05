# World Cup 2026 Winner Prediction

Machine learning pipeline that predicts the 2026 FIFA World Cup from the
Round of 16 onward, built on an XGBoost match-outcome model and Monte Carlo
bracket simulation.

The project freezes all data at **2026-07-03** (end of the Round of 32) and
treats two Round-of-16 ties as unplayed holdouts: **Morocco vs Canada** and
**France vs Paraguay**. Both were predicted correctly (Morocco 68%, France 76%).

## Headline result (20,000 bracket simulations)

| Team | Reach SF | Reach Final | Champion |
|---|---|---|---|
| Argentina | 57.6% | 36.5% | **20.9%** |
| Spain | 49.9% | 32.8% | **20.4%** |
| France | 58.5% | 33.1% | **19.4%** |
| England | 30.9% | 15.8% | 7.6% |
| Brazil | 30.8% | 14.5% | 6.6% |

Full table: [`outputs/championship_probabilities.csv`](outputs/championship_probabilities.csv)

## Data sources

| Source | Content | Access |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | 49,500+ international matches, 1872–present, incl. all WC2026 games | CSV on GitHub |
| Same repo, `shootouts.csv` | Penalty shootout winners | CSV on GitHub |
| [eloratings.net](http://www.eloratings.net) | Current World Football Elo snapshot (validation only) | TSV endpoint |
| FIFA official API | WC2026 match calendar, scores, bracket slots, venues | `api.fifa.com/api/v3` (competition 17, season 285023) |

Raw data is **not** committed (third-party redistribution rights are unclear);
`src/download_data.py` fetches everything in seconds. Historical per-match Elo
is computed internally in a single chronological pass (K scaled by tournament
importance, goal-margin multiplier, +100 home advantage), so all features are
reproducible and leakage-free.

## Method

1. **Features** (19, all strictly pre-match): Elo levels and difference,
   5/10-match points-per-game form, goals for/against, Elo momentum, rest
   days, head-to-head record, tournament category, venue neutrality.
2. **Model**: XGBoost `multi:softprob` over {home win, draw, away win},
   29,990 matches since 1994, time-decay sample weights (10-year half-life),
   temporal validation: train <2023, validate 2023–25 with early stopping,
   test on 2026, refit through the cutoff.
3. **Simulation**: official bracket from the FIFA API. Neutral-venue
   predictions symmetrized over both team orderings; host advantage applied
   for Mexico (Azteca) and the USA (all US venues). Knockout draws resolved
   by a mild Elo-tilted penalty model. 20,000 Monte Carlo runs.

### Performance (out-of-time)

| Model | 2026 test log loss (n=399) | Accuracy |
|---|---|---|
| Constant prior | 1.044 | — |
| Elo-only logistic | 0.887 | 58.4% |
| **XGBoost (this repo)** | **0.869** | **60.4%** |

## Quickstart

```bash
pip install -r requirements.txt
python src/download_data.py     # fetch raw data (~4 MB)
python src/build_dataset.py     # features + team state at cutoff
python src/train_model.py       # train + evaluate XGBoost
python src/simulate.py          # 20k bracket sims -> outputs/
```

## Repository structure

```
src/
  download_data.py   # fetch all raw sources
  build_dataset.py   # Elo engine + feature construction (no leakage)
  train_model.py     # XGBoost training, baselines, temporal evaluation
  simulate.py        # Monte Carlo knockout bracket simulation
outputs/
  championship_probabilities.csv
data/                # created by download_data.py / build_dataset.py (gitignored)
models/              # created by train_model.py (gitignored)
```

## Limitations

- Results-based features only: no xG, shots, or squad market values
  (FBref and Transfermarkt block automated access; FIFA API possession
  fields are sparsely populated).
- Team state is frozen at the cutoff: injuries, suspensions, and lineup
  news are not captured.
- The penalty shootout model is a mild Elo logistic, not a dedicated model.
- A ~21% favorite is an honest statement about knockout football: the most
  likely single winner is Argentina, but the field is more likely than any
  one team.

## License

Code is MIT licensed. Match data belongs to its respective providers
(see Data sources) and is downloaded at build time, not redistributed.
