"""End-to-end Dual-GAT Fairness-aware OD forecasting package."""

from .data import ODBundle, ODWindowDataset, build_chronological_datasets, load_od_bundle
from .dual_gat import DualRoleStaticGAT
from .fair_mamba import FUSION_CHOICES, FairnessAwareMamba, FusionOperator
from .losses import FairODObjective, od_fairness_loss
from .metrics import fair_od_metrics
from .model import FairODForecastModel

__all__ = [
    "ODBundle",
    "ODWindowDataset",
    "build_chronological_datasets",
    "load_od_bundle",
    "DualRoleStaticGAT",
    "FUSION_CHOICES",
    "FusionOperator",
    "FairnessAwareMamba",
    "FairODObjective",
    "od_fairness_loss",
    "fair_od_metrics",
    "FairODForecastModel",
]
