import numpy as np
from chemprop.nn.metrics import MetricRegistry, Metric
from torch import Tensor
import torch

@MetricRegistry.register("score")
class ScoreMetric(Metric):
     def forward(
        self,
        preds: Tensor,
        targets: Tensor,
        mask: Tensor,
        weights: Tensor,
        lt_mask: Tensor,
        gt_mask: Tensor,
    ) -> Tensor:
        normalized_rmse = torch.sqrt(((targets - preds) ** 2).mean()) / (targets.max() - targets.min())
        correct_ratio = ((IC50_to_pIC50(targets) - IC50_to_pIC50(preds)).abs() <= 0.5).float().mean()
        return 0.5 * (1 - min(normalized_rmse, 1) + correct_ratio)

def pIC50_to_IC50(pic50_values):
    """Convert pIC50 values to IC50 (nM)."""
    return 10 ** (9 - pic50_values)

def IC50_to_pIC50(ic50_values):
    """Convert IC50 values to pIC50 (nM)."""
    if isinstance(ic50_values, torch.Tensor):
        return 9 - torch.log10(ic50_values)
    else:
        return 9 - np.log10(ic50_values)