# FairOD

FairOD is a fairness-aware framework for origin-destination (OD) flow prediction in urban mobility. It addresses prediction disparities across sensitive attribute groups in origin and destination roles, temporal bias drift, and asynchronous bias propagation across OD flows.

## Method Overview

The framework has three components: Dual-GAT learns distinct origin and destination representations, fairness-aware Mamba models temporal dynamics with a fair state-transition constraint, and an asynchronous bias module estimates how historically connected OD pairs affect current predictions. The training objective jointly optimizes prediction error, origin- and destination-side fairness, transition fairness, and bias-impact estimation.

The paper evaluates FairOD on three real-world datasets—NYC Taxi, Chicago Taxi, and Shanghai Metro—and reports improved OD fairness while maintaining predictive performance.

## Code Structure

- `fairod/model.py`: FairOD forecasting model, OD-pair encoder, and prediction head.
- `fairod/dual_gat.py`: Graph attention encoder with distinct origin and destination roles.
- `fairod/fair_mamba.py`, `fairod/temporal.py`: Fairness-aware temporal modeling and transition constraints.
- `fairod/asynchronous_bias.py`: Modeling of historical OD bias impacts.
- `fairod/data.py`: OD data loading, window construction, and chronological splitting.




