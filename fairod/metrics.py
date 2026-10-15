"""Evaluation metrics for OD accuracy and side-specific group fairness."""

from __future__ import annotations

import numpy as np


class FairODMetricAccumulator:
    """Accumulate the existing FairOD metrics across batches with bounded memory."""

    def __init__(
        self, sensitive_attribute: np.ndarray, mape_epsilon: float = 1.0
    ) -> None:
        self.sensitive = np.asarray(sensitive_attribute)
        if self.sensitive.ndim != 1:
            raise ValueError("sensitive_attribute must contain one label per node")
        if not np.any(self.sensitive == 0) or not np.any(self.sensitive == 1):
            raise ValueError("both advantaged (0) and disadvantaged (1) groups are required")
        if mape_epsilon <= 0:
            raise ValueError("mape_epsilon must be positive")
        self.mape_epsilon = float(mape_epsilon)
        self.count = 0
        self.abs_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.ape_sum = 0.0
        self.target_abs_sum = 0.0
        self.origin_ape_sums = {0: 0.0, 1: 0.0}
        self.destination_ape_sums = {0: 0.0, 1: 0.0}
        self.origin_counts = {0: 0, 1: 0}
        self.destination_counts = {0: 0, 1: 0}
        self.wape_masks = self._wape_masks()
        self.wape_numerators = {name: 0.0 for name in self.wape_masks}
        self.wape_denominators = {name: 0.0 for name in self.wape_masks}

    def _wape_masks(self) -> dict[str, np.ndarray]:
        sensitive = self.sensitive
        known_pairs = (sensitive[:, None] >= 0) & (sensitive[None, :] >= 0)
        return {
            "origin_0": known_pairs & (sensitive[:, None] == 0),
            "origin_1": known_pairs & (sensitive[:, None] == 1),
            "destination_0": known_pairs & (sensitive[None, :] == 0),
            "destination_1": known_pairs & (sensitive[None, :] == 1),
        }

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape or prediction.ndim != 3:
            raise ValueError("prediction and target must have matching [samples,N,N] shapes")
        if prediction.shape[1:] != (self.sensitive.size, self.sensitive.size):
            raise ValueError("sensitive_attribute must contain one label per node")

        error = prediction - target
        absolute_error = np.abs(error)
        absolute_target = np.abs(target)
        ape = absolute_error / np.maximum(absolute_target, self.mape_epsilon)
        self.count += int(target.size)
        self.abs_error_sum += float(absolute_error.sum(dtype=np.float64))
        self.squared_error_sum += float(np.square(error).sum(dtype=np.float64))
        self.ape_sum += float(ape.sum(dtype=np.float64))
        self.target_abs_sum += float(absolute_target.sum(dtype=np.float64))

        for group in (0, 1):
            origin_values = ape[:, self.sensitive == group, :]
            destination_values = ape[:, :, self.sensitive == group]
            self.origin_ape_sums[group] += float(origin_values.sum(dtype=np.float64))
            self.destination_ape_sums[group] += float(destination_values.sum(dtype=np.float64))
            self.origin_counts[group] += int(origin_values.size)
            self.destination_counts[group] += int(destination_values.size)

        for name, mask in self.wape_masks.items():
            self.wape_numerators[name] += float(absolute_error[:, mask].sum(dtype=np.float64))
            self.wape_denominators[name] += float(absolute_target[:, mask].sum(dtype=np.float64))

    def compute(self) -> dict[str, float | None]:
        if self.count == 0:
            raise ValueError("cannot compute metrics before any batches are accumulated")
        origin_mape = {
            group: self.origin_ape_sums[group] / self.origin_counts[group]
            for group in (0, 1)
        }
        destination_mape = {
            group: self.destination_ape_sums[group] / self.destination_counts[group]
            for group in (0, 1)
        }

        def wape(name: str) -> float | None:
            denominator = self.wape_denominators[name]
            return (
                self.wape_numerators[name] / denominator
                if denominator > 0.0
                else None
            )

        origin_group_0_wape = wape("origin_0")
        origin_group_1_wape = wape("origin_1")
        destination_group_0_wape = wape("destination_0")
        destination_group_1_wape = wape("destination_1")
        return {
            "mae": self.abs_error_sum / self.count,
            "rmse": float(np.sqrt(self.squared_error_sum / self.count)),
            "mape": self.ape_sum / self.count,
            "wape": (
                self.abs_error_sum / self.target_abs_sum
                if self.target_abs_sum > 0.0
                else None
            ),
            "origin_mape_gap": abs(origin_mape[0] - origin_mape[1]),
            "destination_mape_gap": abs(destination_mape[0] - destination_mape[1]),
            "origin_wape_gap": (
                abs(origin_group_0_wape - origin_group_1_wape)
                if origin_group_0_wape is not None and origin_group_1_wape is not None
                else None
            ),
            "destination_wape_gap": (
                abs(destination_group_0_wape - destination_group_1_wape)
                if destination_group_0_wape is not None and destination_group_1_wape is not None
                else None
            ),
        }


def fair_od_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sensitive_attribute: np.ndarray,
    mape_epsilon: float = 1.0,
) -> dict[str, float | None]:


    accumulator = FairODMetricAccumulator(sensitive_attribute, mape_epsilon)
    accumulator.update(prediction, target)
    return accumulator.compute()
