

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class DuplexGraph:


    amplitude_mask: torch.Tensor
    amplitude_weight: torch.Tensor
    phase_mask: torch.Tensor
    phase_weight: torch.Tensor
    relation_labels: torch.Tensor
    connection_target: torch.Tensor
    weighted_hermitian_target: torch.Tensor

    def to(self, device: torch.device | str) -> "DuplexGraph":
        return DuplexGraph(**{name: value.to(device) for name, value in vars(self).items()})


def build_duplex_graph(
    adjacency: torch.Tensor,
    phase_mode: str = "weighted_imbalance",
    eps: float = 1e-8,
) -> DuplexGraph:


    adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square [N,N]")
    adjacency = adjacency.clamp_min(0)
    n = adjacency.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=adjacency.device)
    forward = adjacency > 0
    reverse = forward.T
    connected = forward | reverse

    labels = torch.full((n, n), 3, dtype=torch.long, device=adjacency.device)
    labels[forward & ~reverse] = 0
    labels[~forward & reverse] = 1
    labels[forward & reverse] = 2
    labels[eye] = 3

    symmetric_flow = adjacency + adjacency.T
    amplitude_weight = torch.log1p(symmetric_flow)
    positive = amplitude_weight[amplitude_weight > 0]
    scale = positive.mean() if positive.numel() else adjacency.new_tensor(1.0)
    amplitude_weight = amplitude_weight / scale.clamp_min(eps)
    amplitude_weight[eye] = 1.0

    if phase_mode == "binary":

        phase_weight = reverse.float() - forward.float()
    elif phase_mode == "weighted_imbalance":

        phase_weight = (adjacency.T - adjacency) / symmetric_flow.clamp_min(eps)
    else:
        raise ValueError(f"Unsupported phase_mode: {phase_mode}")
    phase_weight[eye] = 1.0

    amplitude_mask = connected | eye
    phase_mask = connected | eye
    connection_target = connected.float()
    connection_target[eye] = 0.0


    magnitude = torch.log1p(symmetric_flow)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()
    theta_out_minus_in = -phase_weight
    theta_out_minus_in[eye] = 0.0
    weighted_h = torch.complex(
        magnitude * torch.cos(math.pi * theta_out_minus_in / 2),
        magnitude * torch.sin(math.pi * theta_out_minus_in / 2),
    )
    weighted_h[eye] = 0
    return DuplexGraph(
        amplitude_mask=amplitude_mask,
        amplitude_weight=amplitude_weight,
        phase_mask=phase_mask,
        phase_weight=phase_weight,
        relation_labels=labels,
        connection_target=connection_target,
        weighted_hermitian_target=weighted_h,
    )


class DenseGraphAttention(nn.Module):


    def __init__(self, dim: int, heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_center = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_neighbor = nn.Parameter(torch.empty(heads, self.head_dim))
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.attn_center)
        nn.init.xavier_uniform_(self.attn_neighbor)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        n = features.shape[0]
        hidden = self.proj(features).view(n, self.heads, self.head_dim).permute(1, 0, 2)
        center = (hidden * self.attn_center[:, None, :]).sum(-1).unsqueeze(-1)
        neighbor = (hidden * self.attn_neighbor[:, None, :]).sum(-1).unsqueeze(-2)
        logits = F.leaky_relu(center + neighbor, negative_slope=0.2)
        abs_weight = edge_weight.abs().clamp_min(1e-8)
        logits = logits + abs_weight.log().unsqueeze(0)
        logits = logits.masked_fill(~mask.unsqueeze(0), torch.finfo(logits.dtype).min)
        attention = self.dropout(torch.softmax(logits, dim=-1))
        sign = edge_weight.sign()
        aggregated = torch.einsum("huv,uv,hvd->hud", attention, sign, hidden)
        aggregated = aggregated.permute(1, 0, 2).reshape(n, self.dim)
        return self.out_proj(aggregated)


class DuplexLayer(nn.Module):


    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.amplitude = DenseGraphAttention(dim, heads, dropout)
        self.phase = DenseGraphAttention(dim, heads, dropout)
        self.amplitude_cross = DenseGraphAttention(dim, heads, dropout)
        self.phase_cross = DenseGraphAttention(dim, heads, dropout)
        self.amplitude_norm = nn.LayerNorm(dim)
        self.phase_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        amplitude: torch.Tensor,
        phase: torch.Tensor,
        graph: DuplexGraph,
        fuse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        amplitude_update = self.amplitude(
            amplitude, graph.amplitude_mask, graph.amplitude_weight
        )
        phase_update = self.phase(phase, graph.phase_mask, graph.phase_weight)
        if fuse:

            amplitude_update = amplitude_update + self.amplitude_cross(
                phase, graph.amplitude_mask, graph.amplitude_weight
            )
            phase_update = phase_update + self.phase_cross(
                amplitude, graph.phase_mask, graph.phase_weight
            )
        amplitude = self.amplitude_norm(amplitude + self.dropout(F.relu(amplitude_update)))
        phase = self.phase_norm(phase + self.dropout(F.relu(phase_update)))
        return amplitude, phase


class DuplexStaticEncoder(nn.Module):


    def __init__(
        self,
        num_nodes: int,
        dim: int = 64,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        fusion_layer: int | None = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.num_nodes = num_nodes
        self.dim = dim
        self.fusion_layer = layers // 2 if fusion_layer is None else fusion_layer
        self.initial_amplitude = nn.Parameter(torch.randn(num_nodes, dim) * 0.05)
        self.initial_phase = nn.Parameter(torch.randn(num_nodes, dim) * 0.05)
        self.layers = nn.ModuleList(
            [DuplexLayer(dim, heads, dropout) for _ in range(layers)]
        )

    def forward(
        self, graph: DuplexGraph
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        amplitude = self.initial_amplitude
        phase = self.initial_phase
        for index, layer in enumerate(self.layers):
            amplitude, phase = layer(
                amplitude, phase, graph, fuse=index == self.fusion_layer
            )

        phase = torch.tanh(phase)
        complex_embedding = torch.polar(amplitude.abs(), math.pi * phase / 2)

        complex_embedding = complex_embedding * amplitude.sign().detach()
        return amplitude, phase, complex_embedding

    @staticmethod
    def conjugate_roles(complex_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:


        source = torch.view_as_real(complex_embedding).flatten(-2)
        target = torch.view_as_real(complex_embedding.conj().resolve_conj()).flatten(-2)
        return source, target
