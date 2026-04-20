"""Aviation Safety Network (ASN) dataset loader.

Takes the labeled_aviation_reports_YYYY.csv files from the repo, filters to the
28 ADREP categories we train on, constructs the input text as
    narrative + "  [SEP]  " + phase + "  [SEP]  " + aircraft type
and produces stratified train/val/test splits (68/12/20 by default, matching the
report's split declaration).

Also emits per-row parent labels derived from the child_to_parent map, so the
hierarchical heads can be trained with auxiliary parent supervision.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from ..models.heads import load_taxonomy


DEFAULT_TEXT_COLS = ["narrative", "phase", "aircraft_type", "Type"]
DEFAULT_LABEL_COL = "category_code"


def _join_text_row(row: pd.Series, cols: List[str]) -> str:
    parts = []
    for col in cols:
        val = row.get(col)
        if val is None:
            continue
        s = str(val).strip()
        if not s or s.lower() in {"nan", "unknown", "none"}:
            continue
        parts.append(s)
    return " [SEP] ".join(parts)


def load_asn_frame(
    csv_paths: List[str | Path],
    label_col: str = DEFAULT_LABEL_COL,
    text_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load one or more ASN CSVs and return a unified frame with `text` and `label`."""
    text_cols = text_cols or DEFAULT_TEXT_COLS
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p, encoding="utf-8", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1", low_memory=False)
        # Keep only columns we care about (tolerant to extras/missing)
        for c in (*text_cols, label_col):
            if c not in df.columns:
                df[c] = ""
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["text"] = df.apply(lambda r: _join_text_row(r, text_cols), axis=1)
    df = df[df["text"].str.split().str.len() >= 5].copy()
    df = df.rename(columns={label_col: "label"})
    df["label"] = df["label"].fillna("UNK").astype(str).str.strip().str.upper()
    return df[["text", "label"]]


def filter_to_taxonomy(
    df: pd.DataFrame,
    taxonomy_path: str | Path,
    min_samples_per_class: int = 30,
) -> Tuple[pd.DataFrame, List[str], List[str], torch.Tensor]:
    """Drop rows whose label isn't in the 28-child taxonomy and rare classes."""
    child_names, parent_names, c2p = load_taxonomy(taxonomy_path)
    keep = set(child_names)
    df = df[df["label"].isin(keep)].copy()
    counts = df["label"].value_counts()
    too_small = counts[counts < min_samples_per_class].index.tolist()
    if too_small:
        df = df[~df["label"].isin(too_small)].copy()
        # Rebuild child list preserving order
        child_names = [c for c in child_names if c not in too_small]
        # Reindex child_to_parent correspondingly
        reord = load_taxonomy(taxonomy_path)[2].tolist()
        original = load_taxonomy(taxonomy_path)[0]
        keep_idx = [i for i, c in enumerate(original) if c in child_names]
        c2p = torch.tensor([reord[i] for i in keep_idx], dtype=torch.long)
    return df.reset_index(drop=True), child_names, parent_names, c2p


def stratified_split(
    df: pd.DataFrame,
    train_ratio: float = 0.68,
    val_ratio: float = 0.12,
    test_ratio: float = 0.20,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    train_df, tmp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=seed,
        stratify=df["label"],
    )
    val_frac = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        tmp_df, test_size=(1.0 - val_frac), random_state=seed,
        stratify=tmp_df["label"],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


class AviationDataset(Dataset):
    """Encodes text with a tokenizer on the fly."""

    def __init__(
        self,
        texts: List[str],
        child_ids: List[int],
        parent_ids: List[int],
        tokenizer,
        max_length: int = 256,
    ) -> None:
        self.texts = texts
        self.child_ids = child_ids
        self.parent_ids = parent_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.child_ids[idx], dtype=torch.long),
            "parent_labels": torch.tensor(self.parent_ids[idx], dtype=torch.long),
        }


def build_splits(
    csv_paths: List[str | Path],
    taxonomy_path: str | Path,
    tokenizer,
    max_length: int = 256,
    min_samples_per_class: int = 30,
    seed: int = 42,
) -> Dict[str, object]:
    """One-stop builder: CSVs -> taxonomy filter -> stratified split -> tokenized datasets."""
    df = load_asn_frame(csv_paths)
    df, child_names, parent_names, c2p = filter_to_taxonomy(
        df, taxonomy_path, min_samples_per_class=min_samples_per_class,
    )
    child_to_idx = {c: i for i, c in enumerate(child_names)}
    df["child_id"] = df["label"].map(child_to_idx).astype(int)
    df["parent_id"] = df["child_id"].map(lambda i: int(c2p[i].item())).astype(int)

    train_df, val_df, test_df = stratified_split(df, seed=seed)

    def _mk(split_df):
        return AviationDataset(
            texts=split_df["text"].tolist(),
            child_ids=split_df["child_id"].tolist(),
            parent_ids=split_df["parent_id"].tolist(),
            tokenizer=tokenizer,
            max_length=max_length,
        )

    return {
        "train": _mk(train_df),
        "val": _mk(val_df),
        "test": _mk(test_df),
        "child_names": child_names,
        "parent_names": parent_names,
        "child_to_parent": c2p,
        "class_counts": torch.tensor(
            [int((train_df["child_id"] == i).sum()) for i in range(len(child_names))],
            dtype=torch.long,
        ),
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
    }
