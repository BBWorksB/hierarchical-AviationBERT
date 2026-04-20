"""Domain-adaptive MLM pretraining on aviation narratives.

This is the real Aviation-BERT pretraining loop — not the noise-perturbed
placeholder in the original notebook. It takes a base BERT checkpoint and
continues pretraining on aviation narratives using standard HuggingFace
DataCollatorForLanguageModeling, then saves the result to disk so the classifier
can later fine-tune from that checkpoint.

Usage:
    python -m hierarchical_aviation_bert.pretrain_mlm \
        --data_csvs labeled_aviation_reports_*.csv aviTdata.csv \
        --out aviation_bert_ckpt/ \
        --epochs 1 --batch 16
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


def load_texts(csv_paths: List[str]) -> List[str]:
    texts = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        if "narrative" in df.columns:
            col = "narrative"
        else:
            # Fallback: pick the widest text column
            text_cols = [c for c in df.columns if df[c].dtype == object]
            col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
        texts.extend(df[col].dropna().astype(str).tolist())
    # Drop empties and very short strings
    texts = [t.strip() for t in texts if len(t.strip().split()) >= 5]
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csvs", nargs="+", required=True)
    ap.add_argument("--base_model", default="bert-base-uncased")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--mlm_prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--fp16", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(args.out, exist_ok=True)

    texts = load_texts(args.data_csvs)
    logger.info("Loaded %d narratives for MLM pretraining.", len(texts))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForMaskedLM.from_pretrained(args.base_model)

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=args.max_length,
        )

    hfds = Dataset.from_dict({"text": texts})
    hfds = hfds.map(tokenize_fn, batched=True, remove_columns=["text"])
    hfds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_prob,
    )

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=args.fp16 and torch.cuda.is_available(),
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=100,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hfds,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    logger.info("Aviation-BERT checkpoint saved to %s", args.out)


if __name__ == "__main__":
    main()
