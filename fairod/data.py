

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ODBundle:


    od: np.ndarray
    timestamps: np.ndarray
    location_ids: np.ndarray
    economic_vulnerability: np.ndarray
    sensitive_attribute: np.ndarray
    economic_attribute_valid: np.ndarray
    window_seconds: int | None = None
    time_valid: np.ndarray | None = None


def load_od_bundle(path: str | Path) -> ODBundle:
    archive = np.load(Path(path), allow_pickle=False)
    required = {
        "od",
        "timestamps",
        "location_ids",
        "economic_vulnerability",
        "sensitive_attribute",
        "economic_attribute_valid",
    }
    missing = required - set(archive.files)
    if missing:
        raise ValueError(f"OD archive is missing arrays: {sorted(missing)}")
    od = archive["od"]
    if od.ndim != 3 or od.shape[1] != od.shape[2]:
        raise ValueError(f"Expected OD shape [T,N,N], got {od.shape}")
    return ODBundle(
        od=od,
        timestamps=archive["timestamps"],
        location_ids=archive["location_ids"],
        economic_vulnerability=archive["economic_vulnerability"],
        sensitive_attribute=archive["sensitive_attribute"],
        economic_attribute_valid=archive["economic_attribute_valid"],
        window_seconds=(
            int(archive["window_seconds"])
            if "window_seconds" in archive.files
            else None
        ),
        time_valid=(archive["time_valid"] if "time_valid" in archive.files else None),
    )


def load_chicago_bundle(path: str | Path) -> ODBundle:


    data_dir = Path(path)
    od = np.load(data_dir / "od.npy", allow_pickle=False)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    static = np.load(data_dir / "static.npz", allow_pickle=False)
    node_ids = static["node_ids"].astype(np.int64)
    sensitive_attribute = static["groups"].astype(np.int64)

    with (data_dir / "community_income_labels.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source:
        rows = {
            int(row["community_area_id"]): row
            for row in csv.DictReader(source)
        }
    try:
        economic_vulnerability = np.asarray(
            [float(rows[int(node_id)]["under_25000_share"]) for node_id in node_ids],
            dtype=np.float32,
        )
        csv_groups = np.asarray(
            [int(rows[int(node_id)]["group"]) for node_id in node_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"Chicago income labels are missing area/field {exc}") from exc

    if od.ndim != 3 or od.shape[1] != od.shape[2]:
        raise ValueError(f"Expected Chicago OD shape [T,N,N], got {od.shape}")
    if od.shape[1] != len(node_ids) or sensitive_attribute.shape != node_ids.shape:
        raise ValueError("Chicago OD and static node arrays have incompatible shapes")
    if not np.array_equal(csv_groups, sensitive_attribute):
        raise ValueError("Chicago CSV groups do not match static.npz node ordering")

    start = np.datetime64(metadata["start"], "s")
    window_seconds = int(metadata["window_seconds"])
    timestamps = start + np.arange(len(od)) * np.timedelta64(window_seconds, "s")
    economic_attribute_valid = np.isfinite(economic_vulnerability) & (
        sensitive_attribute >= 0
    )
    return ODBundle(
        od=od,
        timestamps=timestamps,
        location_ids=node_ids,
        economic_vulnerability=economic_vulnerability,
        sensitive_attribute=sensitive_attribute,
        economic_attribute_valid=economic_attribute_valid,
        window_seconds=window_seconds,
    )


def load_shanghai_bundle(path: str | Path) -> ODBundle:


    data_dir = Path(path)
    od = np.load(data_dir / "od.npy", mmap_mode="r", allow_pickle=False)
    timestamps = np.load(data_dir / "timestamps.npy", allow_pickle=False)
    static = np.load(data_dir / "static.npz", allow_pickle=False)
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    groups = static["groups"].astype(np.int64)
    vulnerability = static["economic_vulnerability"].astype(np.float32)
    time_valid_path = data_dir / "time_valid.npy"
    time_valid = (
        np.load(time_valid_path, mmap_mode="r", allow_pickle=False)
        if time_valid_path.exists()
        else None
    )
    return ODBundle(
        od=od,
        timestamps=timestamps,
        location_ids=static["node_ids"].astype(np.int64),
        economic_vulnerability=vulnerability,
        sensitive_attribute=groups,
        economic_attribute_valid=static["economic_attribute_valid"].astype(bool),
        window_seconds=int(metadata["window_seconds"]),
        time_valid=time_valid,
    )


class ODWindowDataset(Dataset):


    def __init__(
        self,
        od: np.ndarray,
        history: int,
        horizon: int,
        target_start: int,
        target_end: int,
        timestamps: np.ndarray | None = None,
        window_seconds: int | None = None,
        time_valid: np.ndarray | None = None,
    ) -> None:
        if history < 1 or horizon < 1:
            raise ValueError("history and horizon must be positive")
        if not 0 <= target_start <= target_end <= len(od):
            raise ValueError("invalid target range")
        self.od = od
        self.history = history
        self.horizon = horizon
        if time_valid is not None and len(time_valid) != len(od):
            raise ValueError("time_valid must contain one value per OD snapshot")
        self.target_indices = []
        for index in range(max(history, target_start), target_end - horizon + 1):
            sample_start = index - history
            sample_end = index + horizon
            if time_valid is not None and not np.all(time_valid[sample_start:sample_end]):
                continue
            if timestamps is not None and window_seconds is not None:
                sample_times = np.asarray(
                    timestamps[sample_start:sample_end]
                ).astype("datetime64[s]").astype(np.int64)
                if len(sample_times) > 1 and not np.all(
                    np.diff(sample_times) == window_seconds
                ):
                    continue
            self.target_indices.append(index)
        if not self.target_indices:
            raise ValueError(
                f"No samples for range [{target_start}, {target_end}); "
                f"history={history}, horizon={horizon}"
            )

    def __len__(self) -> int:
        return len(self.target_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        target_index = self.target_indices[item]
        history = self.od[target_index - self.history : target_index]
        target = self.od[target_index : target_index + self.horizon]
        return (
            torch.from_numpy(np.log1p(history).astype(np.float32)),
            torch.from_numpy(np.log1p(target).astype(np.float32)),
            target_index,
        )


def build_chronological_datasets(
    bundle: ODBundle,
    history: int,
    horizon: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[ODWindowDataset, ODWindowDataset, ODWindowDataset, dict[str, int]]:


    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a positive test split")
    total = len(bundle.od)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    datasets = (
        ODWindowDataset(
            bundle.od, history, horizon, history, train_end,
            timestamps=bundle.timestamps, window_seconds=bundle.window_seconds,
            time_valid=bundle.time_valid,
        ),
        ODWindowDataset(
            bundle.od, history, horizon, train_end, val_end,
            timestamps=bundle.timestamps, window_seconds=bundle.window_seconds,
            time_valid=bundle.time_valid,
        ),
        ODWindowDataset(
            bundle.od, history, horizon, val_end, total,
            timestamps=bundle.timestamps, window_seconds=bundle.window_seconds,
            time_valid=bundle.time_valid,
        ),
    )
    return (*datasets, {"train_end": train_end, "val_end": val_end, "total": total})
