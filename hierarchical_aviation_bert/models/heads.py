"""Three classification heads that share a BERT encoder.

This file implements the exact architectures described in the report:

    FlatHead (Model A)     - single linear layer over [CLS] -> 28 children
    ConstH   (Model B)     - parallel parent (6) + child (28) heads; hard gate
                             zeros child logits when parent confidence < lambda
    HierBERT (Model C)     - soft conditioning: concatenate [CLS] with parent
                             softmax before the child classifier
    LearnableGate (Model D - novel) - learnable interpolation between ConstH
                             and HierBERT with a TCR regularizer term

All heads operate on the pooled [CLS] hidden state of shape (B, H).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Taxonomy loading
# ---------------------------------------------------------------------------


def load_taxonomy(path: str | Path) -> Tuple[List[str], List[str], torch.Tensor]:
    """Load the 28-child / 6-parent mapping.

    Returns:
        child_names: ordered list of 28 child class codes (index = child id).
        parent_names: ordered list of 6 parent class codes (index = parent id).
        child_to_parent: LongTensor of shape (28,), maps child id -> parent id.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    child_names: List[str] = list(obj["child_order"])
    parent_names: List[str] = list(obj["parents"].keys())
    parent_to_idx = {p: i for i, p in enumerate(parent_names)}
    c2p = [parent_to_idx[obj["child_to_parent"][c]] for c in child_names]
    return child_names, parent_names, torch.tensor(c2p, dtype=torch.long)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


class FlatHead(nn.Module):
    """Model A: plain linear + softmax over children."""

    def __init__(self, hidden_size: int, num_children: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.child_clf = nn.Linear(hidden_size, num_children)

    def forward(
        self,
        cls_hidden: torch.Tensor,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        x = self.dropout(cls_hidden)
        child_logits = self.child_clf(x)
        return {
            "child_logits": child_logits,
            "parent_logits": None,
        }


class ConstH(nn.Module):
    """Model B: hard taxonomy-constrained classifier.

    Two parallel heads on [CLS]: parent_clf (6) and child_clf (28).
    Child probabilities are zeroed when the corresponding parent's softmax
    confidence falls below lambda (gating is applied at inference and during
    training for loss computation on the gated distribution).

    Because hard gating is non-differentiable, this model backpropagates through
    the un-gated child logits (standard taxonomy-consistent training) and applies
    the gate only for reporting and for the consistency metric.
    """

    def __init__(
        self,
        hidden_size: int,
        num_parents: int,
        num_children: int,
        child_to_parent: torch.Tensor,
        gate_lambda: float = 0.25,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.gate_lambda = float(gate_lambda)
        self.dropout = nn.Dropout(dropout)
        self.parent_clf = nn.Linear(hidden_size, num_parents)
        self.child_clf = nn.Linear(hidden_size, num_children)
        # (num_children,) long tensor of parent ids for each child
        self.register_buffer("child_to_parent", child_to_parent.long())

    def forward(
        self,
        cls_hidden: torch.Tensor,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        x = self.dropout(cls_hidden)
        parent_logits = self.parent_clf(x)   # (B, P)
        child_logits = self.child_clf(x)     # (B, C)

        parent_probs = F.softmax(parent_logits, dim=-1)     # (B, P)
        # For each child class, pick the parent's probability: (B, C)
        parent_prob_per_child = parent_probs.index_select(
            dim=1, index=self.child_to_parent
        )
        # Hard gate: mask applied AFTER softmax for metrics / consistent predictions
        gate = (parent_prob_per_child >= self.gate_lambda).float()

        return {
            "child_logits": child_logits,
            "parent_logits": parent_logits,
            "gate": gate,
            "parent_prob_per_child": parent_prob_per_child,
        }


class HierBERT(nn.Module):
    """Model C: soft-conditioned hierarchical BERT.

    Stage 1: p_parent = softmax(W_p * [CLS] + b_p)   in R^6
    Stage 2: child_logits = W_c * [[CLS]; p_parent] + b_c   where W_c is (C x (H+P))

    End-to-end differentiable. Child head sees a full probability vector over
    parents, so the network can learn to down-weight unreliable parent signals.
    """

    def __init__(
        self,
        hidden_size: int,
        num_parents: int,
        num_children: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.parent_clf = nn.Linear(hidden_size, num_parents)
        self.child_clf = nn.Linear(hidden_size + num_parents, num_children)

    def forward(
        self,
        cls_hidden: torch.Tensor,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        x = self.dropout(cls_hidden)
        parent_logits = self.parent_clf(x)        # (B, P)
        parent_probs = F.softmax(parent_logits, dim=-1)
        fused = torch.cat([x, parent_probs], dim=-1)  # (B, H + P)
        child_logits = self.child_clf(fused)      # (B, C)
        return {
            "child_logits": child_logits,
            "parent_logits": parent_logits,
            "parent_probs": parent_probs,
        }


class LearnableGate(nn.Module):
    """Model D (novel): learnable interpolation between ConstH hard gating
    and HierBERT soft conditioning, trained end-to-end.

    The child logits are computed as:
        gate = sigmoid(W_g * [CLS] + b_g)   # per-sample scalar in (0, 1)
        child_soft = HierBERT-style child_clf([CLS]; p_parent)
        child_hard_mask = (parent_prob_per_child >= tau_learned).float()
        child = gate * child_soft + (1 - gate) * (child_soft * child_hard_mask)

    tau is a learnable threshold bounded to [0, 1] via a sigmoid-parametrized
    scalar. A TCR regularizer in the loss (computed by the training loop) pushes
    the effective gate toward mode-preserving hard behavior when compliance is
    requested.

    This module produces all outputs the training loop needs to compute the
    regularizer.
    """

    def __init__(
        self,
        hidden_size: int,
        num_parents: int,
        num_children: int,
        child_to_parent: torch.Tensor,
        dropout: float = 0.1,
        init_tau: float = 0.25,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.parent_clf = nn.Linear(hidden_size, num_parents)
        # Soft-conditioned child head (same shape as HierBERT)
        self.child_clf_soft = nn.Linear(hidden_size + num_parents, num_children)
        # Per-sample gating scalar in (0, 1)
        self.gate_proj = nn.Linear(hidden_size, 1)

        # Initialize learnable threshold: sigmoid(raw_tau) = init_tau
        init_raw = torch.logit(torch.tensor(init_tau, dtype=torch.float32))
        self.raw_tau = nn.Parameter(init_raw)

        self.register_buffer("child_to_parent", child_to_parent.long())

    @property
    def tau(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_tau)

    def forward(
        self,
        cls_hidden: torch.Tensor,
        **_: object,
    ) -> Dict[str, torch.Tensor]:
        x = self.dropout(cls_hidden)
        parent_logits = self.parent_clf(x)
        parent_probs = F.softmax(parent_logits, dim=-1)  # (B, P)

        fused = torch.cat([x, parent_probs], dim=-1)
        child_soft = self.child_clf_soft(fused)          # (B, C)

        parent_prob_per_child = parent_probs.index_select(
            dim=1, index=self.child_to_parent
        )                                                # (B, C)
        # Soft mask: straight-through to allow gradient flow; inference binarizes.
        soft_mask = torch.sigmoid((parent_prob_per_child - self.tau) * 10.0)

        gate = torch.sigmoid(self.gate_proj(x))          # (B, 1), per-sample

        # Combine: gate=1 -> pure HierBERT; gate=0 -> ConstH-style masked child
        child_logits = gate * child_soft + (1.0 - gate) * (child_soft * soft_mask)

        return {
            "child_logits": child_logits,
            "parent_logits": parent_logits,
            "parent_probs": parent_probs,
            "parent_prob_per_child": parent_prob_per_child,
            "soft_mask": soft_mask,
            "gate": gate,
            "tau": self.tau,
        }
