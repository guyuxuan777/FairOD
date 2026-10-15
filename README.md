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
## Dataset files

This repository includes the `chicago`, `NYC_taxi`, and `Shanghai_metro` data folders used in the FairOD experiments.

The complete Shanghai MetroFlow OD CSV (`Shanghai_metro/MetroFlow/metroData_ODFlow_10.csv`, about 11.3 GB) and the full processed tensor (`Shanghai_metro/processed/od.npy`, about 4.1 GB) exceed GitHub's per-file storage limits and are not included. Download the corrected public MetroFlow source release from [Figshare](https://doi.org/10.6084/m9.figshare.33436675); the included `Shanghai_metro/prepare_shanghai_dataset.py` describes the FairOD preprocessing. Smaller processed subsets are included under `Shanghai_metro/processed_small_64n_20d` and `Shanghai_metro/processed_small_64n_20d_20min`.

Please cite the MetroFlow dataset paper when using its source data: Sun, P. et al., “Human mobility datasets in the complex metro system of Shanghai,” *Scientific Data*, 12, 1061 (2025), https://doi.org/10.1038/s41597-025-05416-8.