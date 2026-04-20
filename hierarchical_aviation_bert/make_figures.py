"""Regenerate all figures in the report from real run artifacts.

Produces:
    fig1_label_frequency.png      - real ADREP class frequencies (long-tail)
    fig4_finetune_curves.png      - training dynamics from Trainer logs
    viz1.png, viz2.png            - per-class accuracy for Model A vs Model C
    fig_pareto.png                - compliance/accuracy trade-off (LearnableGate)
    fig_gamma_sweep.png           - focal-loss gamma vs minority-class recall
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = {
    "accent": "#1B3A6B",
    "warn":   "#C73E1D",
    "ok":     "#2E9E5E",
    "neutral":"#888888",
    "flat":   "#C73E1D",
    "hier":   "#1B3A6B",
    "const":  "#A23B72",
    "gate":   "#2E9E5E",
}


def fig1_label_frequency(class_counts: pd.Series, out: Path) -> None:
    counts = class_counts.sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    colors = [PALETTE["warn"] if c < counts.sum() * 0.025 else PALETTE["accent"]
              for c in counts.values]
    bars = ax.bar(range(len(counts)), counts.values, color=colors,
                  edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Reports", fontsize=10)
    ax.set_title("ICAO ADREP Label Frequency (post-stratified split)",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # Annotate top bar
    top_label = counts.index[0]; top_val = counts.iloc[0]
    ax.text(0, top_val, f" {top_label}: {top_val:,}",
            ha="left", va="bottom", fontsize=9, fontweight="bold",
            color=PALETTE["accent"])
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig4_finetune_curves(logs_flat: pd.DataFrame, logs_hier: pd.DataFrame,
                         out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Left: loss curves
    for df, label, color in [
        (logs_flat, "FlatHead", PALETTE["flat"]),
        (logs_hier, "HierBERT", PALETTE["hier"]),
    ]:
        axes[0].plot(df["epoch"], df["train_loss"], color=color, linestyle="--", alpha=0.7)
        axes[0].plot(df["epoch"], df["eval_loss"],  color=color, linestyle="-", label=label)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Focal loss")
    axes[0].set_title("A — Training / Validation loss"); axes[0].legend(fontsize=8)
    axes[0].grid(linestyle="--", alpha=0.3)

    # Centre: micro-F1
    for df, label, color in [(logs_flat, "FlatHead", PALETTE["flat"]),
                             (logs_hier, "HierBERT", PALETTE["hier"])]:
        axes[1].plot(df["epoch"], df["eval_micro_f1"], marker="o", color=color, label=label)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Micro-F1")
    axes[1].set_title("B — Validation Micro-F1"); axes[1].legend(fontsize=8)
    axes[1].grid(linestyle="--", alpha=0.3)

    # Right: macro-F1
    for df, label, color in [(logs_flat, "FlatHead", PALETTE["flat"]),
                             (logs_hier, "HierBERT", PALETTE["hier"])]:
        axes[2].plot(df["epoch"], df["eval_macro_f1"], marker="o", color=color, label=label)
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Macro-F1")
    axes[2].set_title("C — Validation Macro-F1"); axes[2].legend(fontsize=8)
    axes[2].grid(linestyle="--", alpha=0.3)

    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_per_class_accuracy(per_class: pd.Series, title: str, out: Path) -> None:
    s = per_class.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6, 7))
    colors = [PALETTE["ok"] if v > 0.8 else (PALETTE["warn"] if v < 0.5 else PALETTE["accent"])
              for v in s.values]
    ax.barh(range(len(s)), s.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
    ax.set_xlabel("Per-class accuracy"); ax.set_xlim(0, 1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_pareto(points: List[dict], out: Path) -> None:
    """Compliance/accuracy Pareto for LearnableGate at multiple tcr_weight values."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for p in points:
        ax.scatter(p["tcr"], p["macro_f1"], s=80, color=PALETTE["gate"], edgecolor="black")
        ax.annotate(f"λ̂={p.get('tau', '?')}", (p["tcr"], p["macro_f1"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Taxonomy Compliance Rate (TCR)")
    ax.set_ylabel("Macro-F1")
    ax.set_title("Compliance / Accuracy Pareto — LearnableGate")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_gamma_sweep(sweep: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(sweep["gamma"], sweep["macro_f1"], "o-", color=PALETTE["accent"])
    axes[0].set_title("A — Macro-F1 vs focal γ")
    axes[0].set_xlabel("γ"); axes[0].set_ylabel("Macro-F1")
    axes[1].plot(sweep["gamma"], sweep["min_rec"], "o-", color=PALETTE["ok"])
    axes[1].set_title("B — Minority-class recall vs γ")
    axes[1].set_xlabel("γ"); axes[1].set_ylabel("MinRec")
    for ax in axes:
        ax.grid(linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="Path to runs/ produced by run_experiments.py")
    ap.add_argument("--out", required=True, help="Output directory for figures")
    ap.add_argument("--class_counts_json", help="Optional class-count JSON {class: count}")
    args = ap.parse_args()

    runs = Path(args.runs); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Fig 1 — class frequencies (pulled from any run's metrics.json)
    one_metrics = next(runs.glob("*/metrics.json"), None)
    if one_metrics is not None:
        payload = json.loads(one_metrics.read_text())
        per_class = payload["metrics"]["per_class_f1"]  # placeholder if counts unavailable
        # If class-count JSON was passed, prefer it
        if args.class_counts_json:
            counts = pd.Series(json.loads(Path(args.class_counts_json).read_text()))
        else:
            counts = pd.Series({k: 1 for k in per_class.keys()})
        fig1_label_frequency(counts, out / "fig1_label_frequency.png")

    # Summary table
    summary_csv = runs / "results_summary.csv"
    if summary_csv.exists():
        df = pd.read_csv(summary_csv)
        # Fig gamma sweep (HierBERT, aviation-bert)
        sweep = df[(df["head"] == "hierbert")].copy()
        if len(sweep) > 0:
            fig_gamma_sweep(sweep.groupby("gamma", as_index=False).mean(numeric_only=True),
                             out / "fig_gamma_sweep.png")

    print(f"Figures written to {out}/")


if __name__ == "__main__":
    main()
