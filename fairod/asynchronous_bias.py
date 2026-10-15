

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AsyncBiasOutput:
    fused_representation: torch.Tensor
    base_representation: torch.Tensor
    candidate_fused_representation: torch.Tensor
    candidate_scores: torch.Tensor
    candidate_lags: torch.Tensor
    candidate_valid: torch.Tensor


class AsynchronousBiasModule(nn.Module):


    def __init__(
        self,
        dim: int,
        candidate_window: int = 2,
        candidate_beam: int = 2,
        selected_candidates: int = 2,
        chunk_size: int = 2048,
        max_lag: int | None = None,
    ) -> None:
        super().__init__()
        # ``max_lag`` is retained as a backwards-compatible alias. The paper's
        # notation is K: the number of previous windows searched in (27)-(29).
        if max_lag is not None:
            candidate_window = max_lag
        if candidate_window < 1:
            raise ValueError("candidate_window must be at least 1")
        self.dim = dim
        self.candidate_window = candidate_window
        self.candidate_beam = candidate_beam
        self.selected_candidates = selected_candidates
        self.chunk_size = chunk_size
        self.interaction = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )  # phi_b in Eq. (30)
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )  # f_s in Eq. (33)
        self.gate = nn.Linear(dim * 2, dim)  # Eq. (35)

    def _candidate_bank(
        self, hidden: torch.Tensor, flow: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build backward direction-continuous candidates for every target origin."""

        batch, steps, nodes, _, dim = hidden.shape
        capacity = self.candidate_window * self.candidate_beam
        # Candidate topology is discrete. Search a detached CPU copy and retain only
        # integer indices here; candidate representations are gathered in one batched
        # operation below so autograd does not record one CopySlices node per candidate.
        flow_for_search = flow.detach().cpu()
        index_shape = (batch, nodes, capacity)
        time_indices = torch.full(index_shape, -1, dtype=torch.long)
        source_indices = torch.zeros(index_shape, dtype=torch.long)
        endpoint_indices = torch.zeros(index_shape, dtype=torch.long)
        valid = torch.zeros(index_shape, dtype=torch.bool)
        lags = torch.zeros(index_shape, dtype=torch.long)
        for batch_index in range(batch):
            for target_origin in range(nodes):
                frontier = [target_origin]
                offset = 0
                for lag in range(1, self.candidate_window + 1):
                    time_index = steps - 1 - lag
                    if time_index < 0 or not frontier:
                        break
                    options = []
                    for endpoint in frontier:
                        values = flow_for_search[batch_index, time_index, :, endpoint]
                        top_count = min(self.candidate_beam, nodes)
                        top_values, top_sources = torch.topk(values, top_count)
                        for value, source in zip(top_values, top_sources):
                            if float(value) > 0:
                                options.append((float(value), int(source), endpoint))
                    options.sort(key=lambda item: item[0], reverse=True)
                    options = options[: self.candidate_beam]
                    next_frontier = []
                    for _, source, endpoint in options:
                        if offset >= capacity:
                            break
                        time_indices[batch_index, target_origin, offset] = time_index
                        source_indices[batch_index, target_origin, offset] = source
                        endpoint_indices[batch_index, target_origin, offset] = endpoint
                        valid[batch_index, target_origin, offset] = True
                        lags[batch_index, target_origin, offset] = lag
                        offset += 1
                        if source not in next_frontier:
                            next_frontier.append(source)
                    frontier = next_frontier

        # Flatten time and directed OD coordinates into one axis, then gather every
        # selected hidden state at once. Gradients flow through this gather to hidden.
        time_indices = time_indices.to(hidden.device)
        source_indices = source_indices.to(hidden.device)
        endpoint_indices = endpoint_indices.to(hidden.device)
        valid = valid.to(hidden.device)
        lags = lags.to(hidden.device)
        linear_indices = (
            time_indices.clamp_min(0) * (nodes * nodes)
            + source_indices * nodes
            + endpoint_indices
        )
        flat_hidden = hidden.reshape(batch, steps * nodes * nodes, dim)
        gather_indices = linear_indices.reshape(batch, nodes * capacity, 1).expand(
            -1, -1, dim
        )
        bank = torch.gather(flat_hidden, dim=1, index=gather_indices).reshape(
            batch, nodes, capacity, dim
        )
        bank = bank.masked_fill(~valid[..., None], 0)
        return bank, valid, lags

    def forward(
        self, hidden_sequence: torch.Tensor, observed_flow: torch.Tensor
    ) -> AsyncBiasOutput:
        batch, _, nodes, _, dim = hidden_sequence.shape
        bank, bank_valid, bank_lags = self._candidate_bank(
            hidden_sequence, observed_flow
        )
        candidate_count = bank.shape[2]
        target = hidden_sequence[:, -1].reshape(batch, nodes * nodes, dim)
        origin_index = torch.arange(nodes * nodes, device=target.device) // nodes

        fused_chunks = []
        candidate_fused_chunks = []
        score_chunks = []
        lag_chunks = []
        valid_chunks = []
        for start in range(0, nodes * nodes, self.chunk_size):
            end = min(start + self.chunk_size, nodes * nodes)
            origins = origin_index[start:end]
            target_chunk = target[:, start:end]
            candidates = bank[:, origins]
            candidate_valid = bank_valid[:, origins]
            candidate_lags = bank_lags[:, origins]
            target_expanded = target_chunk[:, :, None, :].expand_as(candidates)
            interaction = self.interaction(
                torch.cat([candidates, target_expanded], dim=-1)
            )  # Eq. (30)
            scores = self.score(interaction).squeeze(-1)
            scores = scores.masked_fill(~candidate_valid, -1e9)

            select_count = min(self.selected_candidates, candidate_count)
            top_scores, top_index = torch.topk(scores, select_count, dim=-1)
            gather_index = top_index[..., None].expand(-1, -1, -1, dim)
            selected = torch.gather(interaction, 2, gather_index)
            selected_valid = torch.gather(candidate_valid, 2, top_index)
            selected_sum = (selected * selected_valid[..., None]).sum(dim=2)
            selected_count = selected_valid.sum(dim=2, keepdim=True).clamp_min(1)
            asynchronous = selected_sum / selected_count  # Eq. (34)
            gate = torch.sigmoid(
                self.gate(torch.cat([target_chunk, asynchronous], dim=-1))
            )
            fused = gate * target_chunk + (1 - gate) * asynchronous  # Eq. (36)
            # A target OD pair without a direction-continuous predecessor should
            # remain unchanged; otherwise the zero placeholder would attenuate it.
            has_candidate = selected_valid.any(dim=2, keepdim=True)
            fused = torch.where(has_candidate, fused, target_chunk)

            # Candidate-on counterfactuals used to supervise f_s with Eq. (31)-(33).
            individual_gate = torch.sigmoid(
                self.gate(torch.cat([target_expanded, interaction], dim=-1))
            )
            candidate_fused = (
                individual_gate * target_expanded
                + (1 - individual_gate) * interaction
            )
            fused_chunks.append(fused)
            candidate_fused_chunks.append(candidate_fused)
            score_chunks.append(scores)
            lag_chunks.append(candidate_lags)
            valid_chunks.append(candidate_valid)

        return AsyncBiasOutput(
            fused_representation=torch.cat(fused_chunks, dim=1).reshape(
                batch, nodes, nodes, dim
            ),
            base_representation=target.reshape(batch, nodes, nodes, dim),
            candidate_fused_representation=torch.cat(
                candidate_fused_chunks, dim=1
            ),
            candidate_scores=torch.cat(score_chunks, dim=1),
            candidate_lags=torch.cat(lag_chunks, dim=1),
            candidate_valid=torch.cat(valid_chunks, dim=1),
        )
