

from __future__ import annotations

import math
from types import SimpleNamespace

import dgl
import torch
from torch import nn

from Dual_gat_code.code.model import DUPLEX_gat


class ReferenceDuplexAdapter(nn.Module):


    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        heads: int = 4,
        layers: int = 3,
        dropout: float = 0.1,
        fusion: str = "add",
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("DUPLEX hidden_dim must be divisible by heads")
        args = SimpleNamespace(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            head=heads,
            n_layers=layers,
            dr_rate=dropout,
            fusion=fusion,
        )
        self.num_nodes = num_nodes
        self.layers = layers
        self.output_dim = output_dim
        self.encoder = DUPLEX_gat(args)
        # Non-attributed graph initialization follows the DUPLEX paper/code.
        self.initial_amplitude = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.05)
        self.initial_phase = nn.Parameter(torch.randn(num_nodes, input_dim) * 0.05)

    @staticmethod
    def build_dgl_graph(adjacency: torch.Tensor) -> dgl.DGLGraph:


        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency must be square")
        n = adjacency.shape[0]
        union = (adjacency + adjacency.T) > 0
        union.fill_diagonal_(True)
        src, dst = union.nonzero(as_tuple=True)
        graph = dgl.graph((src, dst), num_nodes=n, device=adjacency.device)

        pair_total = adjacency[src, dst] + adjacency[dst, src]
        amplitude_weight = torch.log1p(pair_total)
        positive = amplitude_weight[amplitude_weight > 0]
        scale = positive.mean() if positive.numel() else adjacency.new_tensor(1.0)
        amplitude_weight = amplitude_weight / scale.clamp_min(1e-8)
        phase_weight = (adjacency[src, dst] - adjacency[dst, src]) / pair_total.clamp_min(1e-8)
        self_edge = src == dst
        amplitude_weight[self_edge] = 1.0
        phase_weight[self_edge] = 1.0
        graph.edata["am_exist"] = amplitude_weight
        graph.edata["exist"] = phase_weight
        return graph

    def forward(
        self, graph: dgl.DGLGraph
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        blocks = [graph] * self.layers
        amplitude, phase = self.encoder(
            blocks, self.initial_amplitude, self.initial_phase
        )
        phase = torch.tanh(phase)
        complex_embedding = torch.polar(amplitude.abs(), math.pi * phase / 2)
        complex_embedding = complex_embedding * amplitude.sign().detach()
        origin = torch.view_as_real(complex_embedding).flatten(-2)
        destination = torch.view_as_real(
            complex_embedding.conj().resolve_conj()
        ).flatten(-2)
        return amplitude, phase, complex_embedding, origin, destination
