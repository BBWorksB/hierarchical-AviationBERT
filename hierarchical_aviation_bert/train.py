"""Single-run trainer for the hierarchical classifier.

Example:

    python -m hierarchical_aviation_bert.train \
        --backbone bert-base-uncased \
        --head hierbert \
        --gamma 2.0 \
        --data_csvs labeled_aviation_reports_2000.csv ... \
        --taxonomy hierarchical_aviation_bert/data/icao_parent_map.json \
        --out runs/hierbert_bert_gamma2

Designed to be driven from `run_experiments.py` or from the Colab runner.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Trainer, TrainingArguments

from .data.dataset import build_splits
from .losses.focal import FocalLoss, inverse_frequency_alpha
from .models.classifier import AviationBertClassifier
from .eval import compute_all_metrics


logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    backbone: str = "bert-base-uncased"
    head: str = "hierbert"      # flat | consth | hierbert | learnablegate
    gamma: float = 2.0
    gate_lambda: float = 0.25
    max_length: int = 256
    batch_size: int = 16
    grad_accum: int = 2
    epochs: int = 5
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.10
    seed: int = 42
    data_csvs: List[str] = field(default_factory=list)
    taxonomy: str = "hierarchical_aviation_bert/data/icao_parent_map.json"
    out: str = "runs/default"
    parent_loss_weight: float = 0.3
    tcr_weight: float = 0.1
    fp16: bool = True
    use_focal: bool = True


def _make_compute_metrics(class_counts: np.ndarray, child_to_parent: np.ndarray, child_names: List[str]):
    def compute_metrics(pred) -> Dict[str, float]:
        logits = pred.predictions
        # If classifier returned a tuple (child_logits, parent_logits) pack both.
        if isinstance(logits, tuple):
            child_logits, parent_logits = logits
        else:
            child_logits, parent_logits = logits, None
        y_pred_child = np.argmax(child_logits, axis=1)
        y_pred_parent = np.argmax(parent_logits, axis=1) if parent_logits is not None else None
        m = compute_all_metrics(
            y_true=pred.label_ids,
            y_pred_child=y_pred_child,
            y_pred_parent=y_pred_parent,
            class_counts=class_counts,
            child_to_parent=child_to_parent,
            child_names=child_names,
        )
        # Trainer expects a flat dict of scalars
        flat = {k: v for k, v in m.items() if not isinstance(v, dict)}
        flat["eval_macro_f1"] = flat["macro_f1"]
        return flat
    return compute_metrics


def train_one(args: TrainArgs) -> Dict[str, float]:
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    splits = build_splits(
        csv_paths=args.data_csvs,
        taxonomy_path=args.taxonomy,
        tokenizer=tokenizer,
        max_length=args.max_length,
        seed=args.seed,
    )

    child_names = splits["child_names"]
    parent_names = splits["parent_names"]
    c2p = splits["child_to_parent"]            # torch.LongTensor
    class_counts = splits["class_counts"]      # torch.LongTensor

    model = AviationBertClassifier(
        backbone_name=args.backbone,
        head=args.head,
        num_children=len(child_names),
        num_parents=len(parent_names),
        child_to_parent=c2p,
        gate_lambda=args.gate_lambda,
        parent_loss_weight=args.parent_loss_weight,
        tcr_weight=args.tcr_weight,
    )

    if args.use_focal:
        alpha = inverse_frequency_alpha(class_counts.float())
        child_loss = FocalLoss(gamma=args.gamma, alpha=alpha)
    else:
        child_loss = None  # model will default to plain CE
    parent_loss = None
    model.set_loss_fns(child_loss, parent_loss)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=args.fp16 and torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
        save_total_limit=1,
        seed=args.seed,
    )

    compute_metrics = _make_compute_metrics(
        class_counts=class_counts.numpy(),
        child_to_parent=c2p.numpy(),
        child_names=child_names,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_out = trainer.predict(splits["test"])

    final_metrics = compute_metrics(test_out)
    out_path = Path(args.out) / "metrics.json"
    payload = {
        "config": asdict(args),
        "n_train": splits["n_train"], "n_val": splits["n_val"], "n_test": splits["n_test"],
        "child_names": child_names,
        "parent_names": parent_names,
        "metrics": final_metrics,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)
    return final_metrics


def _cli() -> None:
    p = argparse.ArgumentParser()
    for fld in TrainArgs.__dataclass_fields__.values():
        kw = {}
        if fld.type == bool:
            kw["action"] = "store_true"
        elif fld.type == list or fld.name == "data_csvs":
            kw["nargs"] = "+"
            kw["default"] = []
        else:
            kw["type"] = fld.type
            if fld.default is not None:
                kw["default"] = fld.default
        p.add_argument(f"--{fld.name}", **kw)
    ns = p.parse_args()
    args = TrainArgs(**{k: v for k, v in vars(ns).items() if v is not None})
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_one(args)


if __name__ == "__main__":
    _cli()
