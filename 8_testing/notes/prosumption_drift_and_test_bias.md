# Prosumption seasonality/drift and its link to test-set forecaster bias

Scope: does `prosumption`'s long-run behaviour (seasonality, multi-year drift) explain the
positive forecast bias observed on the sealed test year (2019-07-01..2020-06-30)? Source
data: `1_data/processed/df_full.csv` (full range 2017-11-03..2020-07-09, hourly), bias
figures from `8_testing/results/raw_daily/` (`bias_MW` column). Bullet-form working notes,
not written up -- source material for the eventual write-up.

## 1. Seasonality is very strong

Mean `prosumption` by calendar month, pooled across all years in the dataset:

| Month | Mean (MW) | Std (MW) |
|---|---|---|
| Dec (peak) | 3.237 | 0.931 |
| Jan | 3.220 | 0.993 |
| Feb | 2.880 | 1.163 |
| Nov | 2.927 | 1.018 |
| Mar | 2.526 | 1.321 |
| Oct | 2.292 | 1.135 |
| Apr | 1.765 | 1.343 |
| Sep | 1.758 | 1.185 |
| Aug | 1.421 | 1.211 |
| Jun | 1.382 | 1.211 |
| May | 1.329 | 1.354 |
| Jul (trough) | 1.275 | 1.236 |

- Winter (Dec/Jan) mean 3.228 MW vs. summer (Jun/Jul) mean 1.335 MW -- a **2.42x** ratio.
  Smooth, near-symmetric single-cycle seasonality (not a step change), consistent across
  all three years individually.
- Std is highest in the shoulder months (Mar-May: 1.32-1.35 MW) and lowest at the winter
  extreme (Dec: 0.93 MW) -- this is the same pattern already established in
  `forecaster_tables.ipynb`'s day-to-day-std-vs-CRPS scatter (R^2~0.69-0.71 across all four
  forecasters: CRPS tracks day-to-day prosumption variability almost forecaster-independently).

## 2. Modest multi-year drift, concentrated in spring 2020

- Naive first-6mo-vs-last-6mo comparison (2.88 MW vs 1.97 MW) is misleading -- confounded
  by season (first window winter-heavy, last window includes summer).
- Same-calendar-month year-over-year comparison shows a real but modest decline: full
  calendar-year means **2018 = 2.270 MW -> 2019 = 2.118 MW** (-6.7%).
- Deseasonalized trend (each month's mean minus its own calendar-month climatology, linear
  fit over the ~33-month span): slope **-0.084 MW/year**, total drift **-0.231 MW**
  end-to-end (~10% of the overall mean). Not a smooth monotonic decline, though --
  concentrated almost entirely in one event:
  - **March-May 2020 carry the three largest negative anomalies in the whole series**
    (-0.204, -0.339, -0.252 MW) -- coincides with the UK's first COVID-19 lockdown
    (from 2020-03-23). Plausible real demand-suppression effect, not generic drift.
  - **July 2020 (last month in the dataset) swings to the single largest positive
    anomaly** (+0.351 MW) -- the series ends on a rebound, not a continued slide.

## 3. This drift explains most of the month-to-month test-set bias magnitude

Test year (`TEST_START`/`TEST_END` in `eval_raw.py`) = 2019-07-01..2020-06-30, exactly the
window covered by the anomaly series above. Correlated each test month's mean `bias_MW`
(forecast - realised, pooled across all four families/seeds, from `raw_daily`) against
that month's deseasonalized prosumption anomaly:

| Month | Mean bias (MW) | Prosumption anomaly (MW) |
|---|---|---|
| 2019-07 | +0.136 | -0.122 |
| 2019-08 | +0.233 | -0.162 |
| 2019-09 | +0.186 | -0.102 |
| 2019-10 | -0.032 | +0.076 |
| 2019-11 | -0.019 | -0.017 |
| 2019-12 | +0.034 | -0.111 |
| 2020-01 | +0.072 | -0.121 |
| 2020-02 | +0.053 | +0.010 |
| 2020-03 | +0.162 | -0.204 |
| 2020-04 | +0.192 | -0.339 |
| 2020-05 | +0.186 | -0.252 |
| 2020-06 | -0.039 | +0.060 |

- **r = 0.826, r^2 = 0.682** between mean bias and the negative anomaly (i.e. bias rises
  when actual demand undershoots its own seasonal norm). Monotonic/clean across nearly all
  12 months, not just an artefact of the COVID months -- e.g. Oct 2019 and Jun 2020 both
  correctly flip to slightly *negative* bias in the two months where actual demand ran
  *above* norm.
- Mechanism: forecasters learn "seasonal normal" almost entirely from 2018 training data
  (+2019 H1 validation). A test year that runs below that learned norm produces systematic
  over-prediction (positive bias); a test month above norm produces slight
  under-prediction.
- This also explains the overall test-set bias level: mean monthly bias **+0.097 MW**
  almost exactly matches the test year's mean deseasonalized anomaly of **-0.107 MW**.

## 4. Caveats

- Explains ~68% of month-to-month bias *magnitude* variance -- real and substantial, not
  the full story.
- Does **not** explain the *within-day* structure of the bias (concentrated in daytime
  hours 6-20, steepest 8-17) -- that is a separate, already-established structural effect
  (see the hourly bias graphs in `forecast_visualisations.ipynb` /
  `forecast_characteristics.ipynb` Section 10d), presumably tied to how the exogenous
  solar/temperature features interact with the forecaster architecture during daylight
  hours, sitting on top of this monthly drift rather than replaced by it.
- COVID-19 lockdown (Mar-May 2020) is the single largest identifiable contributor to the
  drift but not the only one -- Aug/Sep 2019 (pre-COVID) also show negative anomalies of
  similar order (-0.162, -0.102 MW) paired with elevated bias (+0.233, +0.186 MW), so the
  bias-vs-anomaly relationship holds independent of COVID specifically.
