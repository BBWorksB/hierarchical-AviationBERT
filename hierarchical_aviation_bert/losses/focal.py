"""Focal loss (Lin et al., 2017) for multi-class classification with class imbalance.

This is the real implementation — not the demo-on-toy-arrays cell from the original
notebook. It plugs directly into HuggingFace Trainer via a custom compute_loss.

Reference:
    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal Loss for Dense Object Detection. ICCV.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss.

    L = - alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t is the model's predicted probability for the ground-truth class
    and alpha_t is an optional per-class weight (typically set to the
    inverse-frequency of that class to address long-tail imbalance).

    Args:
        gamma: focusing parameter. gamma=0 recovers weighted cross-entropy.
               The report claims "weighted CE approximates focal loss as gamma -> 0";
               this class makes gamma a real, tunable hyperparameter so the claim
               can be tested.
        alpha: optional per-class weight tensor of shape (num_classes,). If None,
               all classes receive weight 1.0. Pass inverse-frequency weights for
               long-tail data.
        reduction: 'mean' | 'sum' | 'none'. HuggingFace Trainer expects a scalar,
                   so use 'mean' when integrating with Trainer.
        label_smoothing: optional label smoothing epsilon (applied to cross-entropy
                         before the focal modulation). Default 0.0 = no smoothing.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(f"reduction must be mean|sum|none, got {reduction!r}")
        self.gamma = float(gamma)
        self.reduction = reduction
        self.label_smoothing = float(label_smoothing)
        if alpha is not None:
            # Registered as buffer so it moves with .to(device) but is not a parameter.
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None  # type: ignore[assignment]

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes)
            target: (batch,) with integer class ids
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be (B, C), got {tuple(logits.shape)}")

        log_probs = F.log_softmax(logits, dim=-1)  # (B, C)
        # Per-sample CE (optionally smoothed)
        if self.label_smoothing > 0.0:
            n_classes = logits.size(-1)
            nll = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.mean(dim=-1)
            ce = (1 - self.label_smoothing) * nll + self.label_smoothing * smooth
        else:
            ce = F.nll_loss(log_probs, target, reduction="none")

        # Probability assigned to the true class
        pt = torch.exp(-ce).clamp(min=1e-8, max=1.0 - 1e-8)

        focal_weight = (1.0 - pt).pow(self.gamma)
        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, target)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def inverse_frequency_alpha(
    class_counts: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-class inverse-frequency weights usable as focal-loss alpha.

    alpha_k = N / (K * n_k), mean-normalized to 1.0 so absolute loss magnitude
    is comparable to unweighted CE.
    """
    class_counts = class_counts.float().clamp(min=1.0)
    n_total = class_counts.sum()
    k = class_counts.numel()
    alpha = n_total / (k * class_counts)
    if normalize:
        alpha = alpha / alpha.mean()
    return alpha
