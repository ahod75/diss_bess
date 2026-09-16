# Overview
This repository documents the code used for my UCL Energy Systems and Data Analytics MSc thesis. This thesis explored integrating a dispatchable feeder optimisation problem directly into a probabilistic forecaster's training loop, and evaluated the subsequent retrained forecasters' performance against statistical retraining.

## Quick Rundown
- A dispatchable feeder (DF) aggregates local generation and demand assets, such as domestic demand and PV generation, and uses controllable battery storage to optimise a day-ahead net load schedule.
- The operational performance of the DF depends on forecast quality, but a statistically accurate forecast is not always the best performing one.
- Decision-focused learning (DFL) addresses this by integrating the downstream optimisation objective directly into forecaster training.
- Prior DF work has applied DFL to point forecasts where the DF optimises the day-ahead schedule against an idealised cost function.
- This dissertation extends that to a probabilistic forecaster where the DF optimises against GB imbalance settlement prices (historic single-price imbalance data, used to simulate both single- and dual-price imbalance markets) to see whether improvements made in previous papers are reproducible in a realistic market environment.
- Despite decision-focused retraining altering forecaster behaviour in ways explainable by the market mechanisms forecasters were trained against, it produced no reliable economic benefit when compared to statistical re-training across 5 training seeds.
- This is explainable by two main reasons:
    - Using real imbalance prices removes the stable training direction that the fixed, abstract cost function used in previous papers benefitted from.
    - The simplified single-stage training formulation restricted what DFL could learn as it only used the mean forecast estimate, leaving DFL no mechanism visibility and therefore no mechanism to alter the full distribution of the probabilistic forecaster.

## Data used:
- [Electricity demand data and solar generation data from Plymouth, UK: late 2017 - mid 2020](https://zenodo.org/records/5500457)
- [Elexon API - System Price (single-price imbalance data)](https://bmrs.elexon.co.uk/api-documentation/endpoint/balancing/settlement/system-prices/%7BsettlementDate%7D)
- [Ember Energy - Historical GB Day-ahead prices (in EUR)](https://ember-energy.org/data/european-wholesale-electricity-price-data)
- [Source of historical EUR -> GBP conversion rates to convert Day-ahead to GBP](https://uk.investing.com/currencies/eur-gbp-historical-data)

## Machine learning pipeline:
- Probabilistic forecaster: Quantile GRU forecaster based on [this paper](https://dl.acm.org/doi/10.1145/3447555.3464861).
    - Built using **Pytorch**.
    - This forecaster differs from the reference paper by using strictly monotonic increasing quantile heads, in order to create valid CDF for scenario generation. Based on formulation from [this paper](http://arxiv.org/abs/2605.12762).
- Scenario generation: gaussian copula.
    - Built using **Pytorch**.
    - Scenarios generated using fixed sobol-sequence draw, for stable training gradient and even coverage of possible scenarios.
- Simplified DF optimisation problem (to generate DF schedule)
    - Built using **CVXPY, cxpylayers**.
    - Takes only the mean of the generated scenarios (as oppose to all scenarios individually) to optimise day-ahead schedule for DF.
- Market settlement function(to work out realised costs of schedules under single- and dual-price imbalance markets).
    - Built using **Pytorch**.
    - Uses DF decisions and realised net load to calculate realised costs under each imbalance market.

## Evaluation:
- two-stage stochastic robust DF formulation based on [this paper](https://doi.org/10.1109/TPWRS.2022.3152667).
    - Built using **CVXPY**.
    - Evaluated against GB single-price and stylised dual-price imbalance created using Elexon historical prices.

