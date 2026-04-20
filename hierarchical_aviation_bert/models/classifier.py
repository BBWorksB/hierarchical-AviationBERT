"""Unified classifier wrapping a BERT encoder with one of the four heads.

The model is HuggingFace-Trainer compatible: its forward signature matches what
the Trainer passes (input_ids, attention_mask, labels), and it returns a dict
with `loss` and `logits` (plus per-head extras the training loop can use).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

from .heads import FlatHead, ConstH, HierBERT, LearnableGate


@dataclass
class ClassifierOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    parent_logits: Optional[torch.Tensor] = None
    extras: Optional[Dict[str, torch.Tensor]] = None


HEAD_REGISTRY = {
    "flat":          FlatHead,
    "flathead":      FlatHead,
    "consth":        ConstH,
    "hierbert":      HierBERT,
    "learnablegate": LearnableGate,
}


class AviationBertClassifier(nn.Module):
    """BERT encoder + configurable classification head.

    Args:
        backbone_name: HF model id (e.g. 'bert-base-uncased' or a path to a
                       domain-adaptive pretrained Aviation-BERT checkpoint).
        head: one of {'flat', 'consth', 'hierbert', 'learnablegate'}.
        num_children: number of child classes (28 for ADREP).
        num_parents:  number of parents (6).
        child_to_parent: LongTensor mapping child id -> parent id. Required for
                         consth and learnablegate.
        gate_lambda:  threshold for ConstH.
        parent_loss_weight: scalar weight for the auxiliary parent-classification
                            loss term (0 = no aux loss, 1 = equal weight with child).
        tcr_weight:   for LearnableGate, weight on the TCR regularizer term.
    """

    def __init__(
        self,
        backbone_name: str,
        head: str,
        num_children: int,
        num_parents: int,
        child_to_parent: Optional[torch.Tensor] = None,
        gate_lambda: float = 0.25,
        parent_loss_weight: float = 0.3,
        tcr_weight: float = 0.1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden = self.backbone.config.hidden_size
        self.parent_loss_weight = parent_loss_weight
        self.tcr_weight = tcr_weight

        head_key = head.lower()
        if head_key not in HEAD_REGISTRY:
            raise ValueError(f"Unknown head {head!r}. Options: {sorted(HEAD_REGISTRY)}")

        if head_key in ("flat", "flathead"):
            self.head = FlatHead(hidden, num_children, dropout=dropout)
        elif head_key == "consth":
            assert child_to_parent is not None
            self.head = ConstH(
                hidden, num_parents, num_children,
                child_to_parent=child_to_parent,
                gate_lambda=gate_lambda, dropout=dropout,
            )
        elif head_key == "hierbert":
            self.head = HierBERT(hidden, num_parents, num_children, dropout=dropout)
        elif head_key == "learnablegate":
            assert child_to_parent is not None
            self.head = LearnableGate(
                hidden, num_parents, num_children,
                child_to_parent=child_to_parent, dropout=dropout,
                init_tau=gate_lambda,
            )
        self.head_name = head_key

        # Loss is injected from outside (FocalLoss instance) so we can sweep gamma
        # without rebuilding the model.
        self.child_loss_fn: Optional[nn.Module] = None
        self.parent_loss_fn: Optional[nn.Module] = None

    def set_loss_fns(
        self,
        child_loss_fn: nn.Module,
        parent_loss_fn: Optional[nn.Module] = None,
    ) -> None:
        self.child_loss_fn = child_loss_fn
        self.parent_loss_fn = parent_loss_fn

    def _encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] pooled hidden state; prefer pooler_output if available,
        # else the first token of last_hidden_state.
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cls_h = self._encode(input_ids, attention_mask)
        out = self.head(cls_h)
        child_logits = out["child_logits"]
        parent_logits = out.get("parent_logits")

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            if self.child_loss_fn is None:
                # Default: weighted CE with weight 1 (equivalent to standard CE).
                loss = F.cross_entropy(child_logits, labels)
            else:
                loss = self.child_loss_fn(child_logits, labels)

            if parent_logits is not None and parent_labels is not None:
                if self.parent_loss_fn is None:
                    pl = F.cross_entropy(parent_logits, parent_labels)
                else:
                    pl = self.parent_loss_fn(parent_logits, parent_labels)
                loss = loss + self.parent_loss_weight * pl

            # LearnableGate adds a TCR regularizer term
            if self.head_name == "learnablegate":
                # Encourage parent-prob concentration (so the effective gate behaves
                # consistently). Use negative entropy of parent_probs as a soft TCR proxy.
                parent_probs = out["parent_probs"]
                ent = -(parent_probs * (parent_probs.clamp_min(1e-8)).log()).sum(dim=-1).mean()
                loss = loss + self.tcr_weight * ent

        # HF Trainer compatibility: pack output as a dict with only standard keys
        # (loss, logits). For hierarchical heads we concatenate [child; parent]
        # along the class dimension; compute_metrics splits it back. This avoids
        # the NoneType error from missing parent_logits (FlatHead) and the
        # "too many values to unpack" error from extra output keys.
        if parent_logits is not None:
            combined_logits = torch.cat([child_logits, parent_logits], dim=1)
        else:
            combined_logits = child_logits

        return {
            "loss": loss,
            "logits": combined_logits,
        }
