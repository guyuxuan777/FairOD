"""Build a Chicago-style FairOD bundle from Shanghai MetroFlow + 2017 housing data.

The 12 GB OD CSV is streamed in chunks. Outputs are written under
``Shanghai_metro/processed``. The OD array is an int32 memory-mapped NPY file;
its 10-minute timestamps retain the daily 06:00-23:00 service window. Six dates
identified as incomplete by the MetroFlow paper are omitted. Sample windows
that cross the overnight timestamp gaps are filtered by fairod.data.
"""

from __future__ import annotations

import ast
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METRO_DIR = ROOT / "MetroFlow"
OUTPUT_DIR = ROOT / "processed"
OD_FILE = METRO_DIR / "metroData_ODFlow_10.csv"
STATION_FILE = METRO_DIR / "stationInfo.csv"
HOUSING_FILE = ROOT / "Housepricing_Shanghai.txt"
START_DATE = date(2017, 5, 1)
DAY_COUNT = 123
SLOTS_PER_DAY = 102
SECONDS_PER_SLOT = 600
SERVICE_START_MINUTE = 6 * 60
MATCH_RADIUS_M = 800.0
INCOMPLETE_DATES = {
    "20170504", "20170508", "20170509", "20170616", "20170627", "20170628"
}
CITY_BOUNDS = {"longitude": [120.8, 122.2], "latitude": [30.6, 31.9]}


def bd09_to_gcj02(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = lon - 0.0065
    y = lat - 0.006
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * np.pi * 3000.0 / 180.0)
    theta = np.arctan2(y, x) - 0.000003 * np.cos(x * np.pi * 3000.0 / 180.0)
    return z * np.cos(theta), z * np.sin(theta)


def _transform_lat(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    value = -100 + 2 * x + 3 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    value += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    value += (20 * np.sin(y * np.pi) + 40 * np.sin(y / 3 * np.pi)) * 2 / 3
    value += (160 * np.sin(y / 12 * np.pi) + 320 * np.sin(y * np.pi / 30)) * 2 / 3
    return value


def _transform_lon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    value = 300 + x + 2 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    value += (20 * np.sin(6 * x * np.pi) + 20 * np.sin(2 * x * np.pi)) * 2 / 3
    value += (20 * np.sin(x * np.pi) + 40 * np.sin(x / 3 * np.pi)) * 2 / 3
    value += (150 * np.sin(x / 12 * np.pi) + 300 * np.sin(x / 30 * np.pi)) * 2 / 3
    return value


def gcj02_to_wgs84(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Standard approximate inverse of the GCJ-02 offset, adequate for the 800 m join.
    semi_major = 6378245.0
    eccentricity_sq = 0.00669342162296594323
    x, y = lon - 105.0, lat - 35.0
    dlat, dlon = _transform_lat(x, y), _transform_lon(x, y)
    rad_lat = lat / 180.0 * np.pi
    magic = 1 - eccentricity_sq * np.sin(rad_lat) ** 2
    sqrt_magic = np.sqrt(magic)
    dlat = dlat * 180.0 / (
        (semi_major * (1 - eccentricity_sq)) / (magic * sqrt_magic) * np.pi
    )
    dlon = dlon * 180.0 / (
        semi_major / sqrt_magic * np.cos(rad_lat) * np.pi
    )
    return lon - dlon, lat - dlat


def housing_bd09_to_wgs84(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gcj_lon, gcj_lat = bd09_to_gcj02(lon, lat)
    return gcj02_to_wgs84(gcj_lon, gcj_lat)


def haversine_matrix(
    station_lon: np.ndarray,
    station_lat: np.ndarray,
    house_lon: np.ndarray,
    house_lat: np.ndarray,
) -> np.ndarray:
    earth_radius_m = 6_371_008.8
    lat1 = np.radians(station_lat)[:, None]
    lon1 = np.radians(station_lon)[:, None]
    lat2 = np.radians(house_lat)[None, :]
    lon2 = np.radians(house_lon)[None, :]
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_m * np.arcsin(np.sqrt(np.minimum(1.0, a)))


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums > 0)


def station_attributes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    stations = pd.read_csv(STATION_FILE, encoding="utf-8-sig")
    if "stationID" not in stations or stations["stationID"].duplicated().any():
        raise ValueError("stationInfo.csv must have unique stationID values")
    stations = stations.reset_index(drop=True)
    stations.insert(0, "node_index", np.arange(len(stations), dtype=np.int32))
    stations["stationID"] = stations["stationID"].astype(np.int64)

    houses = pd.read_csv(
        HOUSING_FILE,
        header=None,
        names=["community_name", "housing_price_raw", "latitude_bd09", "longitude_bd09"],
        encoding="utf-8-sig",
    )
    for column in ("housing_price_raw", "latitude_bd09", "longitude_bd09"):
        houses[column] = pd.to_numeric(houses[column], errors="coerce")
    valid = (
        houses[["housing_price_raw", "latitude_bd09", "longitude_bd09"]].notna().all(axis=1)
        & (houses["housing_price_raw"] > 0)
    )
    houses = houses.loc[valid].copy().reset_index(drop=True)
    houses["community_record_id"] = np.arange(len(houses), dtype=np.int32)
    houses["longitude"], houses["latitude"] = housing_bd09_to_wgs84(
        houses["longitude_bd09"].to_numpy(dtype=np.float64),
        houses["latitude_bd09"].to_numpy(dtype=np.float64),
    )
    in_city = (
        houses["longitude"].between(*CITY_BOUNDS["longitude"])
        & houses["latitude"].between(*CITY_BOUNDS["latitude"])
    )
    out_of_city_count = int((~in_city).sum())
    houses = houses.loc[in_city].reset_index(drop=True)

    station_lon = stations["lon"].to_numpy(dtype=np.float64)
    station_lat = stations["lat"].to_numpy(dtype=np.float64)
    if not (
        np.all((station_lon >= 120.8) & (station_lon <= 122.2))
        and np.all((station_lat >= 30.6) & (station_lat <= 31.9))
    ):
        raise ValueError("stationInfo coordinates fall outside the Shanghai bounds")
    distances = haversine_matrix(
        station_lon,
        station_lat,
        houses["longitude"].to_numpy(dtype=np.float64),
        houses["latitude"].to_numpy(dtype=np.float64),
    )
    within = distances <= MATCH_RADIUS_M
    prices = houses["housing_price_raw"].to_numpy(dtype=np.float64)
    counts = within.sum(axis=1).astype(np.int32)
    station_price = np.asarray(
        [np.median(prices[row]) if row.any() else np.nan for row in within],
        dtype=np.float64,
    )
    eligible = np.isfinite(station_price)
    threshold = float(np.median(station_price[eligible]))
    groups = np.full(len(stations), -1, dtype=np.int64)
    groups[eligible & (station_price >= threshold)] = 0
    groups[eligible & (station_price < threshold)] = 1
    valid_stations = np.flatnonzero(eligible)
    vulnerability = np.full(len(stations), np.nan, dtype=np.float32)
    # Higher local prices map to a lower vulnerability percentile.
    sorted_price = pd.Series(station_price[eligible]).rank(method="average", pct=True).to_numpy()
    vulnerability[valid_stations] = (1.0 - sorted_price).astype(np.float32)

    station_to_node = {
        int(station_id): int(node_index)
        for node_index, station_id in enumerate(stations["stationID"])
    }
    adjacency = np.zeros((len(stations), len(stations)), dtype=bool)
    invalid_neighbour_ids = set()
    for node_index, value in enumerate(stations["neighbour"]):
        if pd.isna(value) or not str(value).strip():
            continue
        for neighbor_id in ast.literal_eval(str(value)):
            neighbor_node = station_to_node.get(int(neighbor_id))
            if neighbor_node is None:
                invalid_neighbour_ids.add(int(neighbor_id))
            else:
                adjacency[node_index, neighbor_node] = True
                adjacency[neighbor_node, node_index] = True
    np.fill_diagonal(adjacency, False)

    attributes = stations[["node_index", "stationID", "name", "lon", "lat"]].copy()
    attributes = attributes.rename(
        columns={
            "stationID": "original_station_id",
            "name": "station_name",
            "lon": "longitude",
            "lat": "latitude",
        }
    )
    attributes["housing_price_median"] = station_price
    attributes["housing_record_count"] = counts
    attributes["match_radius_m"] = MATCH_RADIUS_M
    attributes["economic_group"] = groups
    attributes["economic_group_description"] = [
        "economically_advantaged_higher_local_housing_price" if group == 0
        else "economically_disadvantaged_lower_local_housing_price" if group == 1
        else "unknown_no_housing_match"
        for group in groups
    ]
    attributes["economic_attribute_valid"] = eligible
    attributes["economic_vulnerability_percentile"] = vulnerability

    matches_rows = []
    for node_index, house_index in zip(*np.where(within)):
        station_row = stations.iloc[int(node_index)]
        house_row = houses.iloc[int(house_index)]
        matches_rows.append(
            {
                "node_index": int(node_index),
                "station_id": int(station_row.stationID),
                "station_name": station_row["name"],
                "community_record_id": int(house_row.community_record_id),
                "community_name": house_row.community_name,
                "housing_price_raw": float(house_row.housing_price_raw),
                "distance_m": float(distances[node_index, house_index]),
            }
        )
    matches = pd.DataFrame(matches_rows)

    n = len(stations)
    geo_distance = haversine_matrix(station_lon, station_lat, station_lon, station_lat)
    aff_geo = np.exp(-np.square(geo_distance / 1000.0)).astype(np.float32)
    aff_geo[geo_distance > 5000] = 0
    np.fill_diagonal(aff_geo, 1)
    # Flow affinity is filled after the chronological split is known.
    static = {
        "groups": groups,
        "node_ids": stations["stationID"].to_numpy(dtype=np.int64),
        "centroids": np.column_stack([station_lon, station_lat]).astype(np.float32),
        "adjacency": adjacency,
        "aff_geo": aff_geo,
        "aff_ff": np.zeros((n, n), dtype=np.float32),
        "aff_hop": np.zeros((n, n), dtype=np.float32),
        "economic_vulnerability": vulnerability,
        "economic_attribute_valid": eligible.astype(bool),
        "housing_price_median": station_price.astype(np.float32),
        "housing_record_count": counts,
    }
    report = {
        "station_count": int(n),
        "housing_rows_raw": int(len(pd.read_csv(HOUSING_FILE, header=None, usecols=[0], encoding="utf-8-sig"))),
        "housing_rows_valid_coordinates_and_price": int(valid.sum()),
        "housing_rows_inside_city_bounds": int(len(houses)),
        "housing_rows_removed_outside_city_bounds": out_of_city_count,
        "housing_rows_without_price_or_coordinates": int((~valid).sum()),
        "match_radius_m": MATCH_RADIUS_M,
        "matched_stations": int(eligible.sum()),
        "unmatched_stations": int((~eligible).sum()),
        "housing_matches_total": int(within.sum()),
        "matches_per_labeled_station": {
            "minimum": int(counts[eligible].min()),
            "median": float(np.median(counts[eligible])),
            "maximum": int(counts[eligible].max()),
        },
        "station_price_median_threshold": threshold,
        "groups": {
            "0_economically_advantaged_higher_price": int((groups == 0).sum()),
            "1_economically_disadvantaged_lower_price": int((groups == 1).sum()),
            "-1_unknown": int((groups == -1).sum()),
        },
        "station_topology_edges_undirected": int(adjacency.sum() // 2),
        "invalid_neighbor_ids": sorted(invalid_neighbour_ids),
        "coordinate_assumptions": {
            "station_coordinates": "CRS is not declared; local station coordinates are treated as WGS84-compatible based on comparison with public map coordinates (inference)",
            "housing_coordinates": "Baidu Maps coordinates treated as BD-09 and converted approximately to WGS84",
            "housing_coordinate_conversion": "BD-09 -> GCJ-02 -> approximate inverse GCJ-02 -> WGS84",
            "city_bounds_wgs84": CITY_BOUNDS,
        },
    }
    return stations, houses, attributes, matches, static, report


def shortest_path_affinity(adjacency: np.ndarray) -> np.ndarray:
    n = len(adjacency)
    hop = np.zeros((n, n), dtype=np.float32)
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
        hop[source, reachable] = 1.0 / (1.0 + distance[reachable])
    return hop


def make_od_array(
    stations: pd.DataFrame,
    static: dict,
    static_report: dict,
) -> tuple[np.ndarray, dict]:
    station_ids = stations["stationID"].to_numpy(dtype=np.int64)
    id_to_node = {int(value): i for i, value in enumerate(station_ids)}
    incomplete = {datetime.strptime(value, "%Y%m%d").date() for value in INCOMPLETE_DATES}
    all_dates = [START_DATE + timedelta(days=i) for i in range(DAY_COUNT)]
    date_to_original_index = {
        day.strftime("%Y%m%d"): i for i, day in enumerate(all_dates)
    }
    complete_dates = [day for day in all_dates if day not in incomplete]
    date_to_complete = {day: i for i, day in enumerate(complete_dates)}
    n = len(station_ids)
    graph_count = len(complete_dates) * SLOTS_PER_DAY
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    od_path = OUTPUT_DIR / "od.npy"
    od = np.lib.format.open_memmap(
        od_path, mode="w+", dtype=np.int32, shape=(graph_count, n, n)
    )
    od[:] = 0
    od.flush()
    del od
    od = np.load(od_path, mmap_mode="r+", allow_pickle=False)
    slot_sets = [set() for _ in all_dates]
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0
    total_flow = 0
    component_mismatch_rows = 0
    negative_flow_rows = 0
    min_flow = None
    max_flow = 0
    duplicate_rows_in_chunks = 0
    started = datetime.now()
    dtype = {
        "date": str,
        "timeslot": np.int32,
        "originStation": np.int32,
        "destinationStation": np.int32,
        "Flow": np.int32,
        "CFlow": np.int32,
        "HBOFlow": np.int32,
        "NHBFlow": np.int32,
    }
    usecols = list(dtype)
    for chunk_number, chunk in enumerate(
        pd.read_csv(
            OD_FILE,
            encoding="utf-8-sig",
            skipinitialspace=True,
            usecols=usecols,
            dtype=dtype,
            chunksize=1_000_000,
            low_memory=False,
        ),
        start=1,
    ):
        total_rows += len(chunk)
        duplicate_rows_in_chunks += int(
            chunk.duplicated(
                ["date", "timeslot", "originStation", "destinationStation"]
            ).sum()
        )
        flows = chunk["Flow"].to_numpy(dtype=np.int32)
        total_flow += int(flows.sum(dtype=np.int64))
        min_flow = int(flows.min()) if min_flow is None else min(min_flow, int(flows.min()))
        max_flow = max(max_flow, int(flows.max()))
        negative_flow_rows += int((flows < 0).sum())
        component_total = (
            chunk["CFlow"].to_numpy(dtype=np.int64)
            + chunk["HBOFlow"].to_numpy(dtype=np.int64)
            + chunk["NHBFlow"].to_numpy(dtype=np.int64)
        )
        component_mismatch_rows += int((component_total != flows.astype(np.int64)).sum())
        day_indices = chunk["date"].astype(str).map(date_to_original_index)
        if day_indices.isna().any():
            raise ValueError("OD rows contain a date outside the documented range")
        day_indices = day_indices.to_numpy(dtype=np.int16)
        timeslots = chunk["timeslot"].to_numpy(dtype=np.int32)
        origin_nodes = chunk["originStation"].map(id_to_node).to_numpy(dtype=np.int32)
        destination_nodes = chunk["destinationStation"].map(id_to_node).to_numpy(dtype=np.int32)
        if (origin_nodes < 0).any() or (destination_nodes < 0).any():
            raise ValueError("OD rows contain station IDs missing from stationInfo.csv")
        frame = np.full(len(chunk), -1, dtype=np.int32)
        for original_day_index in np.unique(day_indices):
            day = all_dates[int(original_day_index)]
            rows = day_indices == original_day_index
            if day in incomplete:
                dropped_rows += int(rows.sum())
                continue
            compact_day_index = date_to_complete[day]
            day_slots = timeslots[rows] - original_day_index * SLOTS_PER_DAY
            if np.any((day_slots < 0) | (day_slots >= SLOTS_PER_DAY)):
                raise ValueError(f"Unexpected timeslot offset for {day}")
            slot_sets[original_day_index].update(int(value) for value in day_slots)
            frame[rows] = compact_day_index * SLOTS_PER_DAY + day_slots
        keep = frame >= 0
        kept_rows += int(keep.sum())
        pair_linear = origin_nodes[keep].astype(np.int64) * n + destination_nodes[keep]
        work = pd.DataFrame(
            {"frame": frame[keep], "pair": pair_linear, "flow": flows[keep]}
        )
        flat_od = od.reshape(-1)
        for frame_index, rows in work.groupby("frame", sort=False):
            pair_idx = rows["pair"].to_numpy(dtype=np.int64)
            flow_values = rows["flow"].to_numpy(dtype=np.int32)
            # add.at correctly sums duplicate OD keys should any occur across chunks.
            np.add.at(flat_od[int(frame_index) * n * n : (int(frame_index) + 1) * n * n], pair_idx, flow_values)
        if chunk_number % 20 == 0:
            print(
                f"[OD] chunks={chunk_number} rows={total_rows:,} "
                f"kept={kept_rows:,} dropped_incomplete={dropped_rows:,} "
                f"elapsed={(datetime.now()-started).total_seconds()/60:.1f}min",
                flush=True,
            )
    od.flush()

    expected_slots = set(range(SLOTS_PER_DAY))
    for day_index, day in enumerate(all_dates):
        if day in incomplete:
            continue
        if slot_sets[day_index] != expected_slots:
            missing = sorted(expected_slots - slot_sets[day_index])
            extra = sorted(slot_sets[day_index] - expected_slots)
            raise ValueError(f"Unexpected slot coverage for {day}: missing={missing[:8]} extra={extra[:8]}")

    timestamps = np.asarray(
        [
            np.datetime64(
                datetime.combine(day, datetime.min.time())
                + timedelta(minutes=SERVICE_START_MINUTE + slot * 10),
                "s",
            )
            for day in complete_dates
            for slot in range(SLOTS_PER_DAY)
        ],
        dtype="datetime64[s]",
    )
    np.save(OUTPUT_DIR / "timestamps.npy", timestamps, allow_pickle=False)
    time_index_path = OUTPUT_DIR / "time_index.csv"
    with time_index_path.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["time_index", "timestamp", "date", "slot_in_day", "status"],
        )
        writer.writeheader()
        for i, timestamp in enumerate(timestamps):
            ts = timestamp.astype("datetime64[s]").astype(datetime)
            writer.writerow(
                {
                    "time_index": i,
                    "timestamp": ts.isoformat(sep=" "),
                    "date": ts.date().isoformat(),
                    "slot_in_day": (ts.hour * 60 + ts.minute - SERVICE_START_MINUTE) // 10,
                    "status": "complete_observed_service_interval",
                }
            )

    total_windows = len(od)
    train_end = int(total_windows * 0.70)
    val_end = int(total_windows * 0.85)
    # Match ODWindowDataset's split boundaries while rejecting histories spanning overnight.
    target_indices = np.arange(6, total_windows, dtype=np.int64)
    seconds = timestamps.astype("datetime64[s]").astype(np.int64)
    continuity = np.ones(total_windows, dtype=bool)
    continuity[1:] = np.diff(seconds) == SECONDS_PER_SLOT
    valid_sample = np.ones(total_windows, dtype=bool)
    for offset in range(6):
        valid_sample[6:] &= continuity[1 + offset : total_windows - 6 + offset + 1]
    sample_mask = valid_sample[target_indices]
    sample_counts = {
        "train": int(sample_mask[target_indices < train_end].sum()),
        "validation": int(sample_mask[(target_indices >= train_end) & (target_indices < val_end)].sum()),
        "test": int(sample_mask[target_indices >= val_end].sum()),
    }

    # Train-only flow affinity; never use validation or test snapshots here.
    train_flow = np.asarray(od[:train_end].sum(axis=0), dtype=np.float64)
    norms = np.linalg.norm(train_flow, axis=1, keepdims=True)
    normalized = np.divide(train_flow, norms, out=np.zeros_like(train_flow), where=norms > 0)
    static["aff_ff"] = (normalized @ normalized.T).astype(np.float32)
    static["aff_ff"] = np.clip(static["aff_ff"], 0, 1)
    static["A_geo_norm"] = row_normalize(static["aff_geo"]).astype(np.float32)
    static["A_ff_norm"] = row_normalize(static["aff_ff"]).astype(np.float32)
    static["aff_hop"] = shortest_path_affinity(static["adjacency"])
    static["A_hop_norm"] = row_normalize(static["aff_hop"]).astype(np.float32)
    np.savez_compressed(OUTPUT_DIR / "static.npz", **static)
    report = {
        "station_count": int(n),
        "od_rows_raw": int(total_rows),
        "od_rows_kept": int(kept_rows),
        "od_rows_excluded_incomplete_dates": int(dropped_rows),
        "od_duplicate_keys_within_chunks": int(duplicate_rows_in_chunks),
        "od_flow_sum_all_source_rows": int(total_flow),
        "flow_min": int(min_flow),
        "flow_max": int(max_flow),
        "negative_flow_rows": int(negative_flow_rows),
        "flow_component_mismatch_rows": int(component_mismatch_rows),
        "original_dates": DAY_COUNT,
        "complete_dates_kept": len(complete_dates),
        "incomplete_dates_excluded": sorted(INCOMPLETE_DATES),
        "slots_per_service_day": SLOTS_PER_DAY,
        "window_seconds": SECONDS_PER_SLOT,
        "windows": int(total_windows),
        "active_service_hours": "06:00-23:00 local time (102 ten-minute bins per day)",
        "time_split": {
            "train_end": train_end,
            "val_end": val_end,
            "total": int(total_windows),
            "train_samples": sample_counts["train"],
            "validation_samples": sample_counts["validation"],
            "test_samples": sample_counts["test"],
            "history": 6,
            "horizon": 1,
            "ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "continuity_rule": "all 6 history intervals and target must be exactly 10 minutes apart; samples never cross overnight gaps",
        },
        "station_economics": static_report,
    }
    return timestamps, report


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stations, houses, attributes, matches, static, static_report = station_attributes()
    stations_out = stations[["node_index", "stationID", "name", "lon", "lat"]].rename(
        columns={
            "stationID": "original_station_id",
            "name": "station_name",
            "lon": "longitude",
            "lat": "latitude",
        }
    )
    stations_out.to_csv(OUTPUT_DIR / "station_index.csv", index=False, encoding="utf-8-sig")
    attributes.to_csv(
        OUTPUT_DIR / "station_economic_attributes.csv", index=False, encoding="utf-8-sig"
    )
    matches.to_csv(
        OUTPUT_DIR / "housing_station_matches.csv", index=False, encoding="utf-8-sig"
    )
    timestamps, report = make_od_array(stations, static, static_report)
    report.update(
        {
            "dataset": "Shanghai MetroFlow + Shanghai Housing Prices",
            "date_start": "2017-05-01",
            "date_end_exclusive": "2017-09-01",
            "window_seconds": SECONDS_PER_SLOT,
            "data_files": {
                "od": "od.npy",
                "static": "static.npz",
                "timestamps": "timestamps.npy",
                "time_index": "time_index.csv",
                "station_index": "station_index.csv",
                "station_economic_attributes": "station_economic_attributes.csv",
                "housing_station_matches": "housing_station_matches.csv",
            },
            "od_semantics": "directed count of passenger trips; absent station-pair rows in a documented complete sparse time bin mean zero flow",
            "housing_price_source": "2017 AnJuKe residential-estate average price; published source does not specify a unit in the local file description, so only relative station-level rankings are used",
            "label_definition": {
                "group_0": "economically advantaged proxy: station housing-price median >= median over all labeled station medians",
                "group_1": "economically disadvantaged proxy: station housing-price median < median over all labeled station medians",
                "group_minus_1": "unknown: no valid residential complex within 800 m",
                "median_tie_rule": "ties assigned to group 0",
                "fairod_mapping": "group 0 is advantaged; group 1 is disadvantaged; -1 is excluded from group fairness gaps",
            },
            "warnings": [
                "The 2017 housing-price proxy is an area-level context attribute, not a passenger income label.",
                "Station CRS is not declared in the local MetroFlow file; WGS84 compatibility is inferred from a public-coordinate spot check.",
                "Baidu housing coordinates are converted approximately from BD-09 to WGS84 before spatial matching.",
                "All Shanghai housing records outside the documented city bounding box and one row with missing values are excluded.",
                "The OD tensor stores only the 117 complete days; overnight gaps remain explicit in timestamps and are excluded from temporal windows.",
            ],
        }
    )
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_readme(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def write_readme(report: dict) -> None:
    split = report["time_split"]
    groups = report["station_economics"]["groups"]
    text = f"""# Shanghai MetroFlow FairOD dataset

Generated from Shanghai MetroFlow and the local 2017 housing-price file.

- OD tensor: `{report['windows']}` observed 10-minute intervals x `{report['station_count']}` stations x `{report['station_count']}` stations.
- Native service window: 06:00-23:00, 102 bins/day. The six incomplete dates are excluded; overnight timestamp gaps remain explicit.
- Snapshot split boundaries: train `< {split['train_end']}`, validation `< {split['val_end']}`, test thereafter.
- Valid one-step samples using six consecutive 10-minute history intervals: train `{split['train_samples']}`, validation `{split['validation_samples']}`, test `{split['test_samples']}`.
- Station groups: `0=high-price/advantaged ({groups['0_economically_advantaged_higher_price']})`, `1=lower-price/disadvantaged ({groups['1_economically_disadvantaged_lower_price']})`, `-1=unmatched ({groups['-1_unknown']})`.

Each station receives the median housing price of residential estates within 800 m. Group 0 means the station median is at least the median across labeled stations; ties go to group 0. Stations with no match remain `-1`.

`od.npy` is an int32 NPY tensor. `static.npz` follows the Chicago static-data naming pattern. `timestamps.npy` and `time_index.csv` preserve real service timestamps, and the FairOD loader filters windows across overnight gaps. Housing coordinates are attributed to Baidu Maps (BD-09) and approximately converted to WGS84. MetroFlow does not declare the station CRS; station coordinates are treated as WGS84-compatible based on comparison with public map coordinates, so this remains an inference.

The housing-price group is a local economic-context proxy, not passenger income. The source description does not specify the price unit, so only relative prices are used.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
