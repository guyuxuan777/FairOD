

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .asynchronous_bias import AsynchronousBiasModule
from .dual_gat import DualRoleStaticGAT
from .fair_mamba import FairnessAwareMamba


class ODPairEncoder(nn.Module):


    def __init__(self, dual_gat_output_dim: int, model_dim: int) -> None:
        super().__init__()
        role_dim = dual_gat_output_dim * 2
        self.origin_projection = nn.Linear(role_dim, model_dim)      
        self.destination_projection = nn.Linear(role_dim, model_dim)  
        self.pair_fusion = nn.Sequential(                             
            nn.Linear(model_dim * 2, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, model_dim),
        )
        self.flow_projection = nn.Sequential(
            nn.Linear(1, model_dim), nn.GELU(), nn.Linear(model_dim, model_dim)
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(
        self,
        origin_role: torch.Tensor,
        destination_role: torch.Tensor,
        history_log1p: torch.Tensor,
    ) -> torch.Tensor:
        origin = F.gelu(self.origin_projection(origin_role))
        destination = F.gelu(self.destination_projection(destination_role))
        nodes = origin.shape[0]
        origin_pair = origin[:, None, :].expand(nodes, nodes, -1)
        destination_pair = destination[None, :, :].expand(nodes, nodes, -1)
        static_pair = self.pair_fusion(
            torch.cat([origin_pair, destination_pair], dim=-1)
        )
        dynamic_flow = self.flow_projection(history_log1p[..., None])
        return self.norm(static_pair[None, None, ...] + dynamic_flow)


class PredictionHead(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(dim // 2, 4)
        self.network = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.network(representation).squeeze(-1))


class FairODForecastModel(nn.Module):


    def __init__(
        self,
        num_nodes: int,
        dual_gat_input_dim: int = 32,
        dual_gat_hidden_dim: int = 64,
        dual_gat_output_dim: int = 32,
        dual_gat_heads: int = 4,
        dual_gat_layers: int = 3,
        model_dim: int = 64,
        dropout: float = 0.1,
        mobility_context_window: int = 3,
        context_fusion: str = "concat",
        transition_fusion: str = "concat",
        context_bandwidth: float = 1.0,
        transition_samples: int = 128,
        async_candidate_window: int = 2,
        async_candidate_beam: int = 2,
        async_selected_candidates: int = 2,
        async_chunk_size: int = 2048,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.dual_gat = DualRoleStaticGAT(
            num_nodes=num_nodes,
            input_dim=dual_gat_input_dim,
            hidden_dim=dual_gat_hidden_dim,
            output_dim=dual_gat_output_dim,
            heads=dual_gat_heads,
            layers=dual_gat_layers,
            dropout=dropout,
        )
        self.pair_encoder = ODPairEncoder(dual_gat_output_dim, model_dim)
        self.fair_mamba = FairnessAwareMamba(
            dim=model_dim,
            mobility_context_window=mobility_context_window,
            context_fusion=context_fusion,
            transition_fusion=transition_fusion,
            context_bandwidth=context_bandwidth,
            transition_samples=transition_samples,
        )
        self.asynchronous_bias = AsynchronousBiasModule(
            dim=model_dim,
            candidate_window=async_candidate_window,
            candidate_beam=async_candidate_beam,
            selected_candidates=async_selected_candidates,
            chunk_size=async_chunk_size,
        )
        self.prediction_head = PredictionHead(model_dim)

    def static_representations(
        self, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.dual_gat(adjacency)

    def forward(
        self,
        history_log1p: torch.Tensor,
        adjacency: torch.Tensor,
        sensitive_attribute: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if history_log1p.ndim != 4:
            raise ValueError("history_log1p must have shape [B,T,N,N]")
        _, _, nodes, destinations = history_log1p.shape
        if nodes != self.num_nodes or destinations != self.num_nodes:
            raise ValueError("history node dimensions do not match the model")

        amplitude, phase, complex_embedding, origin_role, destination_role = (
            self.static_representations(adjacency)
        )
        pair_sequence = self.pair_encoder(
            origin_role, destination_role, history_log1p
        )
        observed_flow = torch.expm1(history_log1p).clamp_min(0)
        temporal = self.fair_mamba(
            pair_sequence, observed_flow, sensitive_attribute
        )
        hidden_sequence = temporal.hidden_sequence
        context_sequence = temporal.context_sequence
        transition_sequence = temporal.transition_sequence
        fair_transition_loss = temporal.fair_transition_loss
        asynchronous = self.asynchronous_bias(
            hidden_sequence, observed_flow
        )

        prediction = self.prediction_head(asynchronous.fused_representation)
        base_prediction = self.prediction_head(asynchronous.base_representation)
        candidate_prediction = self.prediction_head(
            asynchronous.candidate_fused_representation
        )
        return {
            "prediction": prediction,
            "base_prediction": base_prediction,
            "candidate_prediction": candidate_prediction,
            "candidate_scores": asynchronous.candidate_scores,
            "candidate_lags": asynchronous.candidate_lags,
            "candidate_valid": asynchronous.candidate_valid,
            "fair_transition_loss": fair_transition_loss,
            "hidden_sequence": hidden_sequence,
            "context_sequence": context_sequence,
            "transition_sequence": transition_sequence,
            "amplitude": amplitude,
            "phase": phase,
            "complex_embedding": complex_embedding,
            "origin_embedding": origin_role,
            "destination_embedding": destination_role,
        }
