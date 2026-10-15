

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .duplex import DuplexGraph


def duplex_self_supervised_loss(
    amplitude: torch.Tensor,
    complex_embedding: torch.Tensor,
    graph: DuplexGraph,
    connection_weight: float = 0.1,
    weighted_ham_weight: float = 0.1,
) -> dict[str, torch.Tensor]:


    dim = complex_embedding.shape[-1]
    reconstructed_h = complex_embedding @ complex_embedding.conj().T / math.sqrt(dim)
    prototypes = torch.tensor(
        [1j, -1j, 1 + 0j, 0 + 0j],
        dtype=reconstructed_h.dtype,
        device=reconstructed_h.device,
    )
    logits = -(reconstructed_h[..., None] - prototypes).abs()
    labels = graph.relation_labels
    n = labels.shape[0]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=labels.device)
    flat_labels = labels[off_diagonal]
    counts = torch.bincount(flat_labels, minlength=4).float().clamp_min(1)
    class_weight = counts.sum().sqrt() / counts.sqrt()
    class_weight = class_weight / class_weight.mean()
    direction = F.cross_entropy(logits[off_diagonal], flat_labels, weight=class_weight)

    connection_logits = amplitude @ amplitude.T / math.sqrt(amplitude.shape[-1])
    connection = F.binary_cross_entropy_with_logits(
        connection_logits[off_diagonal], graph.connection_target[off_diagonal]
    )
    connected = off_diagonal & graph.connection_target.bool()
    if connected.any():
        weighted_ham = F.smooth_l1_loss(
            torch.view_as_real(reconstructed_h[connected]),
            torch.view_as_real(graph.weighted_hermitian_target[connected]),
        )
    else:
        weighted_ham = reconstructed_h.real.sum() * 0.0
    total = direction + connection_weight * connection + weighted_ham_weight * weighted_ham
    return {
        "loss": total,
        "direction_loss": direction,
        "connection_loss": connection,
        "weighted_ham_loss": weighted_ham,
    }


def _group_mae_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sensitive_attribute: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:


    sensitive = sensitive_attribute.to(prediction.device)
    absolute_error = (prediction - target).abs()
    origin_a = sensitive[:, None] == 0
    origin_d = sensitive[:, None] == 1
    destination_a = sensitive[None, :] == 0
    destination_d = sensitive[None, :] == 1

    def mean_for(mask: torch.Tensor) -> torch.Tensor:
        expanded = mask
        while expanded.ndim < absolute_error.ndim:
            expanded = expanded.unsqueeze(0)
        expanded = expanded.expand_as(absolute_error)
        return (
            absolute_error.masked_select(expanded).mean()
            if expanded.any()
            else absolute_error.new_zeros(())
        )

    e_oa = mean_for(origin_a)
    e_od = mean_for(origin_d)
    e_da = mean_for(destination_a)
    e_dd = mean_for(destination_d)
    fairness = (e_oa - e_od).abs() + (e_da - e_dd).abs()
    return fairness, {
        "mae_origin_advantaged": e_oa,
        "mae_origin_disadvantaged": e_od,
        "mae_destination_advantaged": e_da,
        "mae_destination_disadvantaged": e_dd,
    }, absolute_error


def od_fairness_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sensitive_attribute: torch.Tensor,
    epsilon: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:


    del epsilon
    fairness, terms, _ = _group_mae_terms(prediction, target, sensitive_attribute)
    return fairness, terms


def _counterfactual_bias_impact(
    base_prediction: torch.Tensor,
    candidate_prediction: torch.Tensor,
    target: torch.Tensor,
    sensitive_attribute: torch.Tensor,
    candidate_lags: torch.Tensor,
    temporal_decay: float,
) -> torch.Tensor:


    batch, nodes, _ = base_prediction.shape
    candidates = candidate_prediction.shape[-1]
    sensitive = sensitive_attribute.to(base_prediction.device)
    base_error = (base_prediction - target).abs()
    target_flat = target.reshape(batch, nodes * nodes, 1)
    candidate_error = (candidate_prediction - target_flat).abs()
    delta_error = candidate_error - base_error.reshape(batch, nodes * nodes, 1)

    origin_group = sensitive.repeat_interleave(nodes)
    destination_group = sensitive.repeat(nodes)
    flat_base = base_error.reshape(batch, nodes * nodes)

    def group_gap_and_update(group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask_a = group == 0
        mask_d = group == 1
        count_a = mask_a.sum().clamp_min(1)
        count_d = mask_d.sum().clamp_min(1)
        mean_a = flat_base[:, mask_a].mean(dim=1)
        mean_d = flat_base[:, mask_d].mean(dim=1)
        gap = mean_a - mean_d
        coefficient = torch.zeros_like(group, dtype=base_prediction.dtype)
        coefficient[mask_a] = 1.0 / count_a
        coefficient[mask_d] = -1.0 / count_d
        update = coefficient[None, :, None] * delta_error
        return gap, update

    origin_gap, origin_update = group_gap_and_update(origin_group)
    destination_gap, destination_update = group_gap_and_update(destination_group)
    base_fairness = origin_gap.abs() + destination_gap.abs()
    candidate_fairness = (
        origin_gap[:, None, None] + origin_update
    ).abs() + (
        destination_gap[:, None, None] + destination_update
    ).abs()
    delta = candidate_fairness - base_fairness[:, None, None]  # Eq. (31)
    lag = candidate_lags.to(delta.dtype).clamp_min(1)
    decay = 1.0 + temporal_decay * (lag - 1.0)
    return delta / decay  # Eq. (32)


class FairODObjective(nn.Module):


    def __init__(
        self,
        lambda_fair: float = 0.1,
        lambda_transition: float = 0.1,
        lambda_score: float = 0.1,
        temporal_decay: float = 0.1,
        mape_epsilon: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_fair = lambda_fair
        self.lambda_transition = lambda_transition
        self.lambda_score = lambda_score
        self.temporal_decay = temporal_decay

        self.mape_epsilon = mape_epsilon

    def forward(
        self,
        output: dict[str, torch.Tensor],
        target: torch.Tensor,
        sensitive_attribute: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        prediction_loss = F.l1_loss(output["prediction"], target)  # Eq. (38)
        fairness_loss, fairness_terms, _ = _group_mae_terms(
            output["prediction"], target, sensitive_attribute
        )
        score_target = _counterfactual_bias_impact(
            output["base_prediction"],
            output["candidate_prediction"],
            target,
            sensitive_attribute,
            output["candidate_lags"],
            self.temporal_decay,
        ).detach()
        valid = output["candidate_valid"]
        if valid.any():
            score_loss = F.mse_loss(
                output["candidate_scores"][valid], score_target[valid]
            )  
        else:
            score_loss = output["candidate_scores"].sum() * 0.0
        transition_loss = output["fair_transition_loss"]  # Eq. (24)
        total = (
            prediction_loss
            + self.lambda_fair * fairness_loss
            + self.lambda_transition * transition_loss
            + self.lambda_score * score_loss
        )  
        return {
            "loss": total,
            "prediction_loss": prediction_loss,
            "fairness_loss": fairness_loss,
            "fair_transition_loss": transition_loss,
            "score_loss": score_loss,
            **fairness_terms,
        }
