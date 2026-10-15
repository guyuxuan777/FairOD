

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _masked_attention(
    transformed: torch.Tensor,
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    mask: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:


    source_score = (transformed * source_vector).sum(dim=-1)
    target_score = (transformed * target_vector).sum(dim=-1)
    score = F.leaky_relu(source_score[:, None] + target_score[None, :], 0.2)
    score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
    attention = torch.softmax(score, dim=-1)
    weighted = attention * edge_weight
    weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return weighted @ transformed


class DualGATHead(nn.Module):


    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.amplitude_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.phase_linear = nn.Linear(input_dim, output_dim, bias=False)
        self.amplitude_source = nn.Parameter(torch.empty(output_dim))
        self.amplitude_target = nn.Parameter(torch.empty(output_dim))
        self.phase_in_source = nn.Parameter(torch.empty(output_dim))
        self.phase_in_target = nn.Parameter(torch.empty(output_dim))
        self.phase_out_source = nn.Parameter(torch.empty(output_dim))
        self.phase_out_target = nn.Parameter(torch.empty(output_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.amplitude_linear.weight)
        nn.init.xavier_uniform_(self.phase_linear.weight)
        for parameter in (
            self.amplitude_source,
            self.amplitude_target,
            self.phase_in_source,
            self.phase_in_target,
            self.phase_out_source,
            self.phase_out_target,
        ):
            nn.init.normal_(parameter, std=1.0 / math.sqrt(parameter.numel()))

    def forward(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        union_mask: torch.Tensor,
        incoming_mask: torch.Tensor,
        outgoing_mask: torch.Tensor,
        amplitude_weight: torch.Tensor,
        incoming_weight: torch.Tensor,
        outgoing_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude_transformed = self.amplitude_linear(amplitude)
        phase_transformed = self.phase_linear(phase)
        amplitude_message = _masked_attention(
            amplitude_transformed,
            self.amplitude_source,
            self.amplitude_target,
            union_mask,
            amplitude_weight,
        )
        incoming_message = _masked_attention(
            phase_transformed,
            self.phase_in_source,
            self.phase_in_target,
            incoming_mask,
            incoming_weight,
        )
        outgoing_message = _masked_attention(
            phase_transformed,
            self.phase_out_source,
            self.phase_out_target,
            outgoing_mask,
            outgoing_weight,
        )

        return F.elu(amplitude_message), F.elu(incoming_message - outgoing_message)


class DualGATBlock(nn.Module):
    """Multi-head Dual-GAT layer with concatenated heads."""

    def __init__(self, input_dim: int, output_dim: int, heads: int) -> None:
        super().__init__()
        if output_dim % heads:
            raise ValueError("Dual-GAT output_dim must be divisible by heads")
        head_dim = output_dim // heads
        self.heads = nn.ModuleList(
            [DualGATHead(input_dim, head_dim) for _ in range(heads)]
        )

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = [head(*args) for head in self.heads]
        return (
            torch.cat([output[0] for output in outputs], dim=-1),
            torch.cat([output[1] for output in outputs], dim=-1),
        )


class DualRoleStaticGAT(nn.Module):


    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        dimensions = [input_dim] + [hidden_dim] * max(layers - 1, 0) + [output_dim]
        self.initial_amplitude = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.05)
        self.initial_phase = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.05)
        self.blocks = nn.ModuleList(
            [
                DualGATBlock(dimensions[index], dimensions[index + 1], heads)
                for index in range(layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _graph_fields(adjacency: torch.Tensor) -> tuple[torch.Tensor, ...]:
        adjacency = adjacency.float().clamp_min(0)
        n = adjacency.shape[0]
        identity = torch.eye(n, dtype=torch.bool, device=adjacency.device)
        union = (adjacency + adjacency.T) > 0
        union = union | identity
        incoming = (adjacency.T > 0) | identity
        outgoing = (adjacency > 0) | identity

        amplitude_weight = torch.log1p(adjacency + adjacency.T)
        amplitude_weight = amplitude_weight.masked_fill(~union, 0)
        amplitude_weight = amplitude_weight + identity.float()
        amplitude_weight = amplitude_weight / amplitude_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        incoming_weight = adjacency.T.masked_fill(~incoming, 0) + identity.float()
        incoming_weight = incoming_weight / incoming_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        outgoing_weight = adjacency.masked_fill(~outgoing, 0) + identity.float()
        outgoing_weight = outgoing_weight / outgoing_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        return union, incoming, outgoing, amplitude_weight, incoming_weight, outgoing_weight

    def forward(
        self, adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be a square matrix")
        fields = self._graph_fields(adjacency)
        amplitude = self.initial_amplitude
        phase = self.initial_phase
        for index, block in enumerate(self.blocks):
            amplitude, phase = block(amplitude, phase, *fields)
            if index + 1 < len(self.blocks):
                amplitude = self.dropout(amplitude)
                phase = self.dropout(phase)

        phase = torch.tanh(phase)
        complex_embedding = torch.polar(
            amplitude.abs(), torch.pi * phase / 2
        ) * amplitude.sign().detach()
        origin = torch.view_as_real(complex_embedding).flatten(-2)
        destination = torch.view_as_real(
            complex_embedding.conj().resolve_conj()
        ).flatten(-2)
        return amplitude, phase, complex_embedding, origin, destination
