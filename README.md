# World Cup 2026 Winner Prediction

Machine learning pipeline that predicts the 2026 FIFA World Cup from the
Round of 16 onward, built on an XGBoost match-outcome model and Monte Carlo
bracket simulation.

The project freezes all data at **2026-07-03** (end of the Round of 32) and
treats two Round-of-16 ties as unplayed holdouts: **Morocco vs Canada** and
**France vs Paraguay**. Both were predicted correctly (Morocco 68%, France 76%).

## Headline results (20,000 bracket simulations each)

Championship probability under three model variants:

| Team | XGBoost (Elo/form) | + xG blend | 2026 stats only |
|---|---|---|---|
| Spain | 20.4% | **20.4%** | **29.4%** |
| Argentina | **20.9%** | 19.9% | 6.5% |
| France | 19.4% | 18.9% | 10.3% |
| Brazil | 6.6% | 7.4% | 12.8% |
| England | 7.6% | 7.8% | 7.3% |
| Canada | 0.4% | 0.4% | 9.4% |

The stats-only column uses nothing but 2026 tournament xG, shots, shots on
target, and possession. The divergence is informative: Spain dominates on
underlying numbers (+1.80 xG diff, 0 goals conceded); Argentina's title odds
rest heavily on pedigree (Elo) rather than tournament chance creation; and
Canada's shot dominance is invisible to Elo-based models.

Full tables: [`outputs/`](outputs/)

## Data sources

| Source | Content | Access |
|---|---|---|
| [martj42/international_results](https://github.com/martj42/international_results) | 49,500+ international matches, 1872–present, incl. all WC2026 games | CSV on GitHub |
| Same repo, `shootouts.csv` | Penalty shootout winners | CSV on GitHub |
| [eloratings.net](http://www.eloratings.net) | Current World Football Elo snapshot (validation only) | TSV endpoint |
| FIFA official API | WC2026 match calendar, scores, bracket slots, venues | `api.fifa.com/api/v3` (competition 17, season 285023) |
| Sofascore | Per-match xG, possession, shots for all 90 WC2026 matches | browser-collected, committed at `data/raw/sofascore_match_stats.csv` |
| [FBref](https://fbref.com) | Squad possession + shooting aggregates (cross-check) | browser-collected, committed at `data/raw/fbref_squad_stats.csv` |

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
4. **xG variant** (`simulate.py --xg 0.5`): blends each team's goals-based
   form features 50/50 with cutoff-safe tournament xG rates from Sofascore
   (per-match data aggregated by `build_tournament_stats.py`, holdout games
   excluded). xG estimates the same latent attack/defence rates with less
   finishing noise. Effect: Spain (tournament-best +1.80 xG diff, 0 goals
   conceded) overtakes Argentina as narrow favourite.
   Output: `outputs/championship_probabilities_xg.csv`.
5. **Stats-only variant** (`simulate.py --stats-only`): discards the
   historical model entirely. `stats_model.py` builds opponent-adjusted
   attack/defence ratings from the 88 pre-cutoff 2026 matches using a
   composite of xG (55%), shots on target (30%), and shots (15%), with
   possession as a multiplicative control factor; matchups become Poisson
   scoring rates. 88 matches is far too few to train XGBoost on without
   overfitting, so this rating approach is the statistically sound way to
   use tournament stats exclusively. No venue effect; penalty shootouts
   are 50/50. Output: `outputs/championship_probabilities_stats.csv`.

### Performance (out-of-time)

| Model | 2026 test log loss (n=399) | Accuracy |
|---|---|---|
| Constant prior | 1.044 | — |
| Elo-only logistic | 0.887 | 58.4% |
| **XGBoost (this repo)** | **0.869** | **60.4%** |

### Does historical data beat tournament stats? (backtest)

`evaluate_blend.py` answers this without leakage: XGBoost trained only on
pre-tournament matches, the stats model rebuilt per match from an expanding
window of prior tournament games, both scored on 64 WC2026 matches:

| Ensemble (stats share w) | Log loss | Accuracy |
|---|---|---|
| **w=0.0 — historical XGB only** | **0.773** | **68.8%** |
| w=0.2 | 0.793 | 68.8% |
| w=0.5 | 0.838 | 65.6% |
| w=1.0 — 2026 stats only | 1.118 | 59.4% |

Historical data wins decisively: 3-5 matches of xG per team is too noisy to
stand alone (the pure stats model scores worse than the constant prior).
The default simulation therefore uses the historical model; `--blend W`
ensembles the stats model in at any weight, and `--stats-only` shows what
the eye-test metrics alone believe.

## Quickstart

```bash
pip install -r requirements.txt
python src/download_data.py            # fetch raw data (~4 MB)
python src/build_dataset.py            # features + team state at cutoff
python src/train_model.py              # train + evaluate XGBoost
python src/build_tournament_stats.py   # aggregate xG/shots/possession
python src/simulate.py                 # 20k bracket sims (default: historical XGB)
python src/simulate.py --xg 0.5        # xG-blended form features
python src/simulate.py --blend 0.2     # ensemble: 80% XGB + 20% 2026 stats
python src/simulate.py --stats-only    # 2026 tournament stats only
python src/evaluate_blend.py           # backtest: XGB vs stats vs blends
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

- xG enters only at simulation time (feature blend); the historical training
  set has no xG, so the model itself is trained on results-based features.
  Squad market values remain unused (Transfermarkt blocks automated access).
- Team state is frozen at the cutoff: injuries, suspensions, and lineup
  news are not captured.
- The penalty shootout model is a mild Elo logistic, not a dedicated model.
- A ~21% favorite is an honest statement about knockout football: the most
  likely single winner is Argentina, but the field is more likely than any
  one team.

## License

Code is MIT licensed. Match data belongs to its respective providers
(see Data sources) and is downloaded at build time, not redistributed.
