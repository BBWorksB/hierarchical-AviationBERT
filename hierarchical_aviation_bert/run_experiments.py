"""Drive the full experimental matrix.

Sweeps over {head} x {backbone} x {gamma}. Writes one metrics.json per run,
then emits a consolidated `results_summary.csv` with the table rows the paper
needs.

Example:
    python -m hierarchical_aviation_bert.run_experiments \
        --data_csvs labeled_aviation_reports_*.csv aviTdata.csv \
        --out_root runs/
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from .train import TrainArgs, train_one


logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csvs", nargs="+", required=True)
    ap.add_argument("--taxonomy",
                    default="hierarchical_aviation_bert/data/icao_parent_map.json")
    ap.add_argument("--out_root", default="runs")
    ap.add_argument("--backbones", nargs="+",
                    default=["bert-base-uncased", "./aviation_bert_ckpt"],
                    help="First entry is the general baseline, second is Aviation-BERT.")
    ap.add_argument("--heads", nargs="+",
                    default=["flat", "consth", "hierbert", "learnablegate"])
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 1.0, 2.0, 5.0])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--quick", action="store_true",
                    help="Subsample for a smoke test (2 epochs, fewer combinations).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)

    if args.quick:
        heads = ["flat", "hierbert"]
        backbones = [args.backbones[0]]
        gammas = [0.0, 2.0]
        epochs = 2
    else:
        heads, backbones, gammas = args.heads, args.backbones, args.gammas
        epochs = args.epochs

    rows = []
    for head, backbone, gamma in itertools.product(heads, backbones, gammas):
        run_name = f"{head}__{Path(backbone).name.replace('/', '_')}__g{gamma}"
        run_dir = out_root / run_name
        if (run_dir / "metrics.json").exists():
            logger.info("Skipping %s (already done)", run_name)
            payload = json.loads((run_dir / "metrics.json").read_text())
        else:
            logger.info("Running %s", run_name)
            targs = TrainArgs(
                backbone=backbone,
                head=head,
                gamma=gamma,
                data_csvs=args.data_csvs,
                taxonomy=args.taxonomy,
                out=str(run_dir),
                epochs=epochs,
                batch_size=args.batch_size,
            )
            train_one(targs)
            payload = json.loads((run_dir / "metrics.json").read_text())

        m = payload["metrics"]
        rows.append({
            "head": head,
            "backbone": backbone,
            "gamma": gamma,
            "macro_f1": m["macro_f1"],
            "micro_f1": m["micro_f1"],
            "accuracy": m["accuracy"],
            "min_rec": m["minority_recall"],
            "tcr": m["tcr"],
        })

    summary = pd.DataFrame(rows).sort_values(["head", "backbone", "gamma"])
    summary_path = out_root / "results_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s", summary_path)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
