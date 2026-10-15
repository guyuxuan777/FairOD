

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class FairMambaOutput:
    hidden_sequence: torch.Tensor
    context_sequence: torch.Tensor
    transition_sequence: torch.Tensor
    fair_transition_loss: torch.Tensor


FUSION_CHOICES = ("concat", "add", "mean", "gated")


class FusionOperator(nn.Module):


    def __init__(
        self, input_dims: tuple[int, ...], output_dim: int, mode: str
    ) -> None:
        super().__init__()
        if mode not in FUSION_CHOICES:
            raise ValueError(
                f"Unsupported fusion mode {mode!r}; choose from {FUSION_CHOICES}"
            )
        self.mode = mode
        self.output_dim = output_dim
        if mode == "concat":
            self.concat_mlp = nn.Sequential(
                nn.Linear(sum(input_dims), output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
        else:
            self.projections = nn.ModuleList(
                [nn.Linear(input_dim, output_dim) for input_dim in input_dims]
            )
            if mode == "gated":
                self.gate_scores = nn.ModuleList(
                    [nn.Linear(output_dim, 1) for _ in input_dims]
                )
            self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        if self.mode == "concat":
            return self.concat_mlp(torch.cat(inputs, dim=-1))
        projected = torch.stack(
            [projection(value) for projection, value in zip(self.projections, inputs)],
            dim=-2,
        )
        if self.mode == "add":
            fused = projected.sum(dim=-2)
        elif self.mode == "mean":
            fused = projected.mean(dim=-2)
        else:
            logits = torch.cat(
                [
                    scorer(projected[..., index, :])
                    for index, scorer in enumerate(self.gate_scores)
                ],
                dim=-1,
            )
            weights = torch.softmax(logits, dim=-1)
            fused = (projected * weights[..., None]).sum(dim=-2)
        return self.output_norm(F.gelu(fused))


class FairnessAwareMamba(nn.Module):


    def __init__(
        self,
        dim: int,
        mobility_context_window: int = 3,
        context_fusion: str = "concat",
        transition_fusion: str = "concat",
        context_bandwidth: float = 1.0,
        transition_samples: int = 128,
    ) -> None:
        super().__init__()
        if mobility_context_window < 1:
            raise ValueError("mobility_context_window must be at least 1")
        self.dim = dim
        self.mobility_context_window = mobility_context_window
        self.context_fusion_mode = context_fusion
        self.transition_fusion_mode = transition_fusion
        self.context_bandwidth = context_bandwidth
        self.transition_samples = transition_samples

        self.norm = nn.LayerNorm(dim)
        # Eq. (16): input-dependent Delta, B and C.
        self.selective_parameters = nn.Linear(dim, dim * 3)
        self.a_log = nn.Parameter(torch.zeros(dim))
        self.skip = nn.Parameter(torch.ones(dim))
        # Eq. (19): selectable Fuse(e, x, mu, sigma).
        self.context_fusion = FusionOperator(
            (dim, 1, 1, 1), dim, context_fusion
        )
        # Eq. (21): selectable Fuse(retention, injection, readout).
        self.transition_fusion = FusionOperator(
            (dim, dim, dim), dim, transition_fusion
        )

    def _historical_statistics(
        self, flow: torch.Tensor, step: int
    ) -> tuple[torch.Tensor, torch.Tensor]:


        if step == 0:
            zeros = torch.zeros_like(flow[:, 0])
            return zeros, zeros
        start = max(0, step - self.mobility_context_window)
        preceding_flow = flow[:, start:step]
        return (
            preceding_flow.mean(dim=1),
            preceding_flow.std(dim=1, unbiased=False),
        )

    def _sampled_cross_group_alignment(
        self,
        context: torch.Tensor,
        transition: torch.Tensor,
        sensitive_attribute: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:


        device = context.device
        sensitive = sensitive_attribute.to(device)
        origin_group = sensitive.repeat_interleave(num_nodes)
        destination_group = sensitive.repeat(num_nodes)
        total = context.new_zeros(())
        terms = 0
        for group_vector in (origin_group, destination_group):
            index_a = (group_vector == 0).nonzero(as_tuple=True)[0]
            index_d = (group_vector == 1).nonzero(as_tuple=True)[0]
            sample_count = min(
                self.transition_samples, index_a.numel(), index_d.numel()
            )
            if sample_count == 0:
                continue
            index_a = index_a[
                torch.randperm(index_a.numel(), device=device)[:sample_count]
            ]
            index_d = index_d[
                torch.randperm(index_d.numel(), device=device)[:sample_count]
            ]
            for batch_index in range(context.shape[0]):
                context_a = context[batch_index, index_a]
                context_d = context[batch_index, index_d]
                # Matching indices are discrete; similarity values retain gradients to Fuse_c.
                distances = torch.cdist(context_a, context_d).square()
                nearest = distances.detach().argmin(dim=1)
                chosen_distance = distances[
                    torch.arange(sample_count, device=device), nearest
                ]
                omega = torch.exp(
                    -chosen_distance / max(self.context_bandwidth**2, 1e-8)
                )
                transition_distance = (
                    transition[batch_index, index_a]
                    - transition[batch_index, index_d[nearest]]
                ).square().sum(-1)
                total = total + (omega * transition_distance).mean()
                terms += 1
        return total / max(terms, 1)

    def forward(
        self,
        pair_sequence: torch.Tensor,
        observed_flow: torch.Tensor,
        sensitive_attribute: torch.Tensor,
    ) -> FairMambaOutput:
        if pair_sequence.ndim != 5:
            raise ValueError("pair_sequence must have shape [B,T,N,N,D]")
        batch, steps, nodes, destinations, dim = pair_sequence.shape
        if nodes != destinations or dim != self.dim:
            raise ValueError("FairMamba dimensions do not match")
        pairs = nodes * nodes
        sequence = pair_sequence.reshape(batch, steps, pairs, dim)
        flow = observed_flow.reshape(batch, steps, pairs)
        state = sequence.new_zeros(batch, pairs, dim)
        continuous_a = -torch.exp(self.a_log)

        hidden_steps = []
        context_steps = []
        transition_steps = []
        transition_losses = []
        for step in range(steps):
            current = self.norm(sequence[:, step])
            delta_raw, input_gain, readout_gain = self.selective_parameters(
                current
            ).chunk(3, dim=-1)
            delta = F.softplus(delta_raw).clamp(max=1.0)
            # Zero-order hold discretization used by equations (16)-(17).
            a_bar = torch.exp(delta * continuous_a)
            b_bar = torch.expm1(delta * continuous_a) / continuous_a * input_gain
            retention = a_bar * state
            injection = b_bar * current
            state = retention + injection
            readout = readout_gain * state
            hidden = readout + self.skip * current  # Eq. (18)

            local_mean, local_std = self._historical_statistics(flow, step)
            context = self.context_fusion(  # Eq. (19)
                current,
                flow[:, step, :, None],
                local_mean[:, :, None],
                local_std[:, :, None],
            )
            transition = self.transition_fusion(  # Eq. (21)
                retention, injection, readout
            )
            transition_losses.append(
                self._sampled_cross_group_alignment(
                    context, transition, sensitive_attribute, nodes
                )
            )
            hidden_steps.append(hidden)
            context_steps.append(context)
            transition_steps.append(transition)

        hidden_sequence = torch.stack(hidden_steps, dim=1).reshape(
            batch, steps, nodes, nodes, dim
        )
        context_sequence = torch.stack(context_steps, dim=1).reshape(
            batch, steps, nodes, nodes, dim
        )
        transition_sequence = torch.stack(transition_steps, dim=1).reshape(
            batch, steps, nodes, nodes, dim
        )
        fair_transition_loss = torch.stack(transition_losses).mean()  # Eq. (24)
        return FairMambaOutput(
            hidden_sequence=hidden_sequence,
            context_sequence=context_sequence,
            transition_sequence=transition_sequence,
            fair_transition_loss=fair_transition_loss,
        )
