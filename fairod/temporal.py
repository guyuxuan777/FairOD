

from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


class PortableSelectiveSSM(nn.Module):


    def __init__(
        self,
        dim: int,
        state_dim: int = 8,
        conv_kernel: int = 3,
        expansion: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner = dim * expansion
        self.dim = dim
        self.inner = inner
        self.state_dim = state_dim
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, inner * 2)
        self.depthwise_conv = nn.Conv1d(
            inner,
            inner,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=inner,
        )
        self.param_proj = nn.Linear(inner, inner + state_dim * 2)
        self.a_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1).float())[None, :].repeat(inner, 1))
        self.skip = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        values, gates = self.in_proj(self.norm(inputs)).chunk(2, dim=-1)
        length = values.shape[1]
        values = self.depthwise_conv(values.transpose(1, 2))[..., :length].transpose(1, 2)
        values = F.silu(values)
        delta_raw, b_param, c_param = torch.split(
            self.param_proj(values), [self.inner, self.state_dim, self.state_dim], dim=-1
        )
        delta = F.softplus(delta_raw).clamp(max=1.0)
        transition = -torch.exp(self.a_log)
        state = values.new_zeros(values.shape[0], self.inner, self.state_dim)
        outputs = []
        for step in range(length):
            dt = delta[:, step, :, None]
            decay = torch.exp(dt * transition[None, :, :])
            drive = dt * b_param[:, step, None, :] * values[:, step, :, None]
            state = decay * state + drive
            readout = (state * c_param[:, step, None, :]).sum(-1)
            outputs.append(readout + self.skip * values[:, step])
        output = torch.stack(outputs, dim=1) * F.silu(gates)
        return residual + self.dropout(self.out_proj(output))


class TemporalEncoder(nn.Module):


    def __init__(
        self,
        dim: int,
        backend: str = "portable_ssm",
        layers: int = 2,
        state_dim: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backend = backend
        if backend == "portable_ssm":
            self.encoder = nn.Sequential(
                *[
                    PortableSelectiveSSM(dim, state_dim=state_dim, dropout=dropout)
                    for _ in range(layers)
                ]
            )
        elif backend == "gru":
            self.encoder = nn.GRU(
                dim,
                dim,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
            )
        elif backend == "mamba":

            source_root = Path(__file__).resolve().parents[1] / "mamba-main"
            if source_root.is_dir() and str(source_root) not in sys.path:
                sys.path.insert(0, str(source_root))
            try:
                from mamba_ssm import Mamba
            except Exception as exc:  
                raise RuntimeError(
                    "Official Mamba backend is unavailable. Install its dependencies or "
                    "use backend='portable_ssm'."
                ) from exc
            self.encoder = nn.Sequential(
                *[
                    Mamba(d_model=dim, d_state=state_dim, d_conv=4, expand=2)
                    for _ in range(layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported temporal backend: {backend}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.backend == "gru":
            output, _ = self.encoder(inputs)
            return output
        return self.encoder(inputs)
