# Shanghai MetroFlow Small Experimental Subset (10-minute intervals)

This is a reproducible, balanced subset of `Shanghai_metro/processed` for quick model iteration.

- Stations: 64 total, with 32 Group 0 and 32 Group 1 stations. Stations are selected by deterministic spatial farthest-point sampling within each group. Unknown-label stations are excluded.
- Time range: 2017-05-10 through 2017-05-29 inclusive, a contiguous block of 20 complete service days.
- OD snapshots: 2040 snapshots, each at 10-minute resolution, with shape `[T, 64, 64]`.
- For 20-minute intervals, each matrix is the element-wise sum of two adjacent 10-minute OD matrices.
- The six-history/one-step forecasting split remains chronological (70%/15%/15%). Overnight gaps are retained in timestamps and filtered by the loader.
- The quick-run configuration is `configs/shanghai_fair_od_small.yaml`.

Use this subset for rapid debugging and exploratory experiments. It is not a substitute for final results on all 302 stations and all 11,934 snapshots.
