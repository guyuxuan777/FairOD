# Shanghai MetroFlow FairOD dataset

Generated from Shanghai MetroFlow and the local 2017 housing-price file.

- OD tensor: `11934` observed 10-minute intervals x `302` stations x `302` stations.
- Native service window: 06:00-23:00, 102 bins/day. The six incomplete dates are excluded; overnight timestamp gaps remain explicit.
- Snapshot split boundaries: train `< 8353`, validation `< 10143`, test thereafter.
- Valid one-step samples using six consecutive 10-minute history intervals: train `7861`, validation `1682`, test `1689`.
- Station groups: `0=high-price/advantaged (140)`, `1=lower-price/disadvantaged (139)`, `-1=unmatched (23)`.

Each station receives the median housing price of residential estates within 800 m. Group 0 means the station median is at least the median across labeled stations; ties go to group 0. Stations with no match remain `-1`.

`od.npy` is an int32 NPY tensor. `static.npz` follows the Chicago static-data naming pattern. `timestamps.npy` and `time_index.csv` preserve real service timestamps, and the FairOD loader filters windows across overnight gaps. Housing coordinates are attributed to Baidu Maps (BD-09) and approximately converted to WGS84. MetroFlow does not declare the station CRS; station coordinates are treated as WGS84-compatible based on comparison with public map coordinates, so this remains an inference.

The housing-price group is a local economic-context proxy, not passenger income. The source description does not specify the price unit, so only relative prices are used.

The full `od.npy` tensor (about 4.1 GB) is omitted from this GitHub repository because it exceeds the Git LFS single-file limit. Download the MetroFlow source data from https://doi.org/10.6084/m9.figshare.33436675 and regenerate it with `../prepare_shanghai_dataset.py` if needed.