"""Create a balanced, contiguous Shanghai MetroFlow subset for quick experiments."""

from __future__ import annotations

import csv
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Shanghai_metro" / "processed"
OUTPUT = ROOT / "Shanghai_metro" / "processed_small_64n_20d"
N_PER_GROUP = 32
SERVICE_DAYS = 20
SLOTS_PER_DAY = 102
SECONDS_PER_SLOT = 600


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    totals = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, totals, out=np.zeros_like(matrix), where=totals > 0)


def select_spatially_spread(indices: np.ndarray, coordinates: np.ndarray, count: int) -> list[int]:
    """Deterministic farthest-point sampling within one sensitive group."""
    if len(indices) < count:
        raise ValueError(f"Need {count} stations, found only {len(indices)}")
    points = coordinates[indices].astype(np.float64, copy=True)
    points[:, 0] *= np.cos(np.deg2rad(points[:, 1]))
    center = points.mean(axis=0)
    first = int(np.argmin(np.square(points - center).sum(axis=1)))
    chosen = [first]
    minimum_distance = np.square(points - points[first]).sum(axis=1)
    while len(chosen) < count:
        next_index = int(np.argmax(minimum_distance))
        chosen.append(next_index)
        distance = np.square(points - points[next_index]).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[chosen] = -1
    return [int(indices[index]) for index in chosen]


def first_contiguous_dates(timestamps: np.ndarray) -> tuple[list[str], int, int]:
    date_labels = np.asarray([str(value)[:10] for value in timestamps])
    unique_dates, counts = np.unique(date_labels, return_counts=True)
    date_days = unique_dates.astype("datetime64[D]")
    for offset in range(len(unique_dates) - SERVICE_DAYS + 1):
        block = unique_dates[offset : offset + SERVICE_DAYS]
        days = date_days[offset : offset + SERVICE_DAYS]
        if not np.all(np.diff(days) == np.timedelta64(1, "D")):
            continue
        if np.any(counts[offset : offset + SERVICE_DAYS] != SLOTS_PER_DAY):
            continue
        start = int(np.flatnonzero(date_labels == block[0])[0])
        end = int(np.flatnonzero(date_labels == block[-1])[-1]) + 1
        if end - start != SERVICE_DAYS * SLOTS_PER_DAY:
            continue
        return [str(value) for value in block], start, end
    raise RuntimeError(f"Could not find {SERVICE_DAYS} consecutive complete service dates")


def shortest_path_affinity(adjacency: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    affinity = np.zeros((n, n), dtype=np.float32)
    for source in range(n):
        distance = np.full(n, -1, dtype=np.int32)
        distance[source] = 0
        queue = [source]
        for node in queue:
            for neighbor in np.flatnonzero(adjacency[node]):
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    queue.append(int(neighbor))
        reachable = distance >= 0
        affinity[source, reachable] = 1.0 / (1.0 + distance[reachable])
    return affinity


def subset_csv(
    source_path: Path,
    destination_path: Path,
    selected_nodes: set[int] | None = None,
    old_to_new: dict[int, int] | None = None,
    selected_dates: set[str] | None = None,
    slot_stride: int = 1,
) -> list[dict[str, str]]:
    with source_path.open("r", encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            if selected_nodes is not None and int(row["node_index"]) not in selected_nodes:
                continue
            if selected_dates is not None and row["date"] not in selected_dates:
                continue
            if "slot_in_day" in row and int(row["slot_in_day"]) % slot_stride:
                continue
            if old_to_new is not None:
                row["node_index"] = str(old_to_new[int(row["node_index"])])
            if "time_index" in row:
                row["time_index"] = str(len(rows))
            if "slot_in_day" in row:
                row["slot_in_day"] = str(int(row["slot_in_day"]) // slot_stride)
            rows.append(row)
    with destination_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interval-minutes", type=int, choices=(10, 20), default=10,
        help="output interval; 20-minute bins sum adjacent source 10-minute OD matrices",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="rewrite files in the generated subset directory",
    )
    args = parser.parse_args()
    interval_minutes = args.interval_minutes
    factor = interval_minutes // 10
    output = OUTPUT if factor == 1 else ROOT / "Shanghai_metro" / "processed_small_64n_20d_20min"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    timestamps = np.load(SOURCE / "timestamps.npy", allow_pickle=False)
    selected_dates, time_start, time_end = first_contiguous_dates(timestamps)
    static_archive = np.load(SOURCE / "static.npz", allow_pickle=False)
    static = {key: static_archive[key] for key in static_archive.files}
    groups = static["groups"].astype(np.int64)
    coordinates = static["centroids"].astype(np.float64)

    selected_nodes = []
    for group in (0, 1):
        candidates = np.flatnonzero(groups == group)
        selected_nodes.extend(select_spatially_spread(candidates, coordinates, N_PER_GROUP))
    selected_nodes = np.asarray(sorted(selected_nodes), dtype=np.int64)
    old_to_new = {int(old): new for new, old in enumerate(selected_nodes)}
    selected_node_set = set(old_to_new)

    source_od = np.load(SOURCE / "od.npy", mmap_mode="r", allow_pickle=False)
    output_windows = (time_end - time_start) // factor
    subset_od_path = output / "od.npy"
    expected_od_shape = (output_windows, len(selected_nodes), len(selected_nodes))
    if subset_od_path.exists() and args.overwrite:
        existing_od = np.load(subset_od_path, mmap_mode="r", allow_pickle=False)
        if existing_od.shape != expected_od_shape or existing_od.dtype != np.int32:
            raise ValueError(
                f"Existing OD tensor at {subset_od_path} has incompatible shape/dtype: "
                f"{existing_od.shape}/{existing_od.dtype}"
            )
        del existing_od
    else:
        subset_od = np.lib.format.open_memmap(
            subset_od_path, mode="w+", dtype=np.int32, shape=expected_od_shape
        )
        output_block_size = 32
        for output_start in range(0, output_windows, output_block_size):
            output_end = min(output_start + output_block_size, output_windows)
            block_start = time_start + output_start * factor
            block_end = time_start + output_end * factor
            node_slice = source_od[block_start:block_end][:, selected_nodes, :]
            node_slice = node_slice[:, :, selected_nodes]
            if factor > 1:
                node_slice = node_slice.reshape(
                    output_end - output_start, factor, len(selected_nodes), len(selected_nodes)
                ).sum(axis=1, dtype=np.int32)
            subset_od[output_start:output_end] = node_slice
        subset_od.flush()
        del subset_od

    subset_timestamps = timestamps[time_start:time_end:factor]
    np.save(output / "timestamps.npy", subset_timestamps, allow_pickle=False)

    # Preserve station attributes, but make all graph arrays internally consistent
    # with the induced 64-station network and this subset's training period.
    n_source = len(groups)
    subset_static: dict[str, np.ndarray] = {}
    for key, value in static.items():
        if value.ndim == 2 and value.shape == (n_source, n_source):
            subset_static[key] = value[np.ix_(selected_nodes, selected_nodes)].copy()
        elif value.ndim >= 1 and value.shape[0] == n_source:
            subset_static[key] = value[selected_nodes].copy()
        else:
            subset_static[key] = value.copy()
    adjacency = subset_static["adjacency"].astype(bool)
    subset_static["aff_hop"] = shortest_path_affinity(adjacency)
    subset_static["A_geo_norm"] = row_normalize(subset_static["aff_geo"])
    train_end = int(output_windows * 0.70)
    subset_od_readonly = np.load(subset_od_path, mmap_mode="r", allow_pickle=False)
    train_flow = subset_od_readonly[:train_end].sum(axis=0, dtype=np.float64)
    norms = np.linalg.norm(train_flow, axis=1, keepdims=True)
    normalized_flow = np.divide(train_flow, norms, out=np.zeros_like(train_flow), where=norms > 0)
    subset_static["aff_ff"] = np.clip(normalized_flow @ normalized_flow.T, 0, 1).astype(np.float32)
    subset_static["A_ff_norm"] = row_normalize(subset_static["aff_ff"])
    subset_static["A_hop_norm"] = row_normalize(subset_static["aff_hop"])
    np.savez_compressed(output / "static.npz", **subset_static)

    subset_csv(
        SOURCE / "time_index.csv",
        output / "time_index.csv",
        selected_dates=set(selected_dates),
        slot_stride=factor,
    )
    station_rows = subset_csv(
        SOURCE / "station_index.csv",
        output / "station_index.csv",
        selected_nodes=selected_node_set,
        old_to_new=old_to_new,
    )
    subset_csv(
        SOURCE / "station_economic_attributes.csv",
        output / "station_economic_attributes.csv",
        selected_nodes=selected_node_set,
        old_to_new=old_to_new,
    )
    housing_rows = subset_csv(
        SOURCE / "housing_station_matches.csv",
        output / "housing_station_matches.csv",
        selected_nodes=selected_node_set,
        old_to_new=old_to_new,
    )

    subset_group_counts = {
        "0_economically_advantaged_higher_price": int(np.count_nonzero(subset_static["groups"] == 0)),
        "1_economically_disadvantaged_lower_price": int(np.count_nonzero(subset_static["groups"] == 1)),
        "-1_unknown": int(np.count_nonzero(subset_static["groups"] < 0)),
    }
    metadata = {
        "dataset": f"Shanghai MetroFlow balanced small subset ({interval_minutes}-minute intervals)",
        "source_dataset": "Shanghai MetroFlow + Shanghai Housing Prices",
        "source_directory": "Shanghai_metro/processed",
        "subset": {
            "purpose": "quick experimental iteration; not a replacement for full-data final evaluation",
            "station_selection": "32 stations from each known group, selected by deterministic spatial farthest-point sampling",
            "source_node_indices": selected_nodes.tolist(),
            "source_station_ids": [int(value) for value in subset_static["node_ids"]],
            "service_dates": selected_dates,
            "source_interval_minutes": 10,
            "output_interval_minutes": interval_minutes,
            "temporal_aggregation": (
                "sum of adjacent source OD matrices" if factor > 1 else "none; source 10-minute matrices retained"
            ),
        },
        "station_count": int(len(selected_nodes)),
        "group_counts": subset_group_counts,
        "od_shape": [int(output_windows), int(len(selected_nodes)), int(len(selected_nodes))],
        "od_dtype": "int32",
        "window_seconds": SECONDS_PER_SLOT * factor,
        "windows": int(output_windows),
        "date_start": selected_dates[0],
        "date_end_exclusive": str(np.datetime64(selected_dates[-1], "D") + np.timedelta64(1, "D")),
        "slots_per_service_day": SLOTS_PER_DAY // factor,
        "time_split": {
            "train_end": train_end,
            "val_end": int(output_windows * 0.85),
            "total": int(output_windows),
            "train_ratio": 0.70,
            "validation_ratio": 0.15,
            "test_ratio": 0.15,
            "history": 6,
            "horizon": 1,
            "continuity_rule": f"all 6 history intervals and the target must be exactly {interval_minutes} minutes apart; overnight gaps are excluded",
        },
        "station_economics": {
            "station_count": int(len(selected_nodes)),
            "matched_stations": int(np.count_nonzero(subset_static["economic_attribute_valid"])),
            "unmatched_stations": int(np.count_nonzero(~subset_static["economic_attribute_valid"])),
            "housing_matches_total": int(len(housing_rows)),
            "station_price_median_threshold": 60316.5,
            "groups": subset_group_counts,
        },
        "label_definition": {
            "group_0": "housing-price median at or above the source labeled-station median",
            "group_1": "housing-price median below the source labeled-station median",
            "group_minus_1": "unknown because no matched residential estate was available",
            "subset_selection": "balanced across groups 0 and 1; no unknown-label stations selected",
        },
        "data_files": {
            "od": "od.npy",
            "static": "static.npz",
            "timestamps": "timestamps.npy",
            "time_index": "time_index.csv",
            "station_index": "station_index.csv",
            "station_economic_attributes": "station_economic_attributes.csv",
            "housing_station_matches": "housing_station_matches.csv",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    config_name = (
        "configs/shanghai_fair_od_small.yaml"
        if factor == 1
        else "configs/shanghai_fair_od_small_20min.yaml"
    )
    readme = f"""# Shanghai MetroFlow Small Experimental Subset ({interval_minutes}-minute intervals)

This is a reproducible, balanced subset of `Shanghai_metro/processed` for quick model iteration.

- Stations: 64 total, with 32 Group 0 and 32 Group 1 stations. Stations are selected by deterministic spatial farthest-point sampling within each group. Unknown-label stations are excluded.
- Time range: {selected_dates[0]} through {selected_dates[-1]} inclusive, a contiguous block of 20 complete service days.
- OD snapshots: {output_windows} snapshots, each at {interval_minutes}-minute resolution, with shape `[T, 64, 64]`.
- For 20-minute intervals, each matrix is the element-wise sum of two adjacent 10-minute OD matrices.
- The six-history/one-step forecasting split remains chronological (70%/15%/15%). Overnight gaps are retained in timestamps and filtered by the loader.
- The quick-run configuration is `{config_name}`.

Use this subset for rapid debugging and exploratory experiments. It is not a substitute for final results on all 302 stations and all 11,934 snapshots.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"output": str(output), "metadata": metadata, "station_rows": len(station_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
