# Hierarchical AviationBERT — Team 30 (11-785)

Code supporting *Hierarchical Transformer Architectures for ICAO ADREP Aviation Incident Classification*.

## What's in here

- `models/` — the three classification heads on a shared BERT encoder:
  - `FlatHead` — single-layer softmax baseline (Model A in the paper).
  - `ConstH` — hard taxonomy-constrained (parent + child heads, threshold λ).
  - `HierBERT` — soft-conditioned hierarchical head (concatenates soft parent probabilities with [CLS] before the child classifier).
  - `LearnableGate` — novel learnable interpolation between ConstH hard gating and HierBERT soft conditioning, trained end-to-end with a TCR regularizer. Addresses the compliance/accuracy trade-off flagged by TA1.
- `losses/focal.py` — actual focal loss (Lin et al. 2017) with per-class α and configurable γ, not a demo on toy arrays.
- `data/icao_parent_map.json` — 28 ICAO ADREP occurrence categories grouped into 6 parents per the ICAO/CAST-ICAO CICTT taxonomy.
- `data/dataset.py` — loads ASN narratives, applies the same `text = narrative + phase + aircraft type` concatenation the report describes, produces stratified 68/12/20 splits.
- `train.py` — single-run trainer (HuggingFace Trainer + custom head + custom loss).
- `eval.py` — metrics: Macro/Micro-F1, per-class F1, minority-class recall, TCR (taxonomy compliance rate).
- `run_experiments.py` — drives the full matrix {FlatHead, ConstH, HierBERT, LearnableGate} × {BERT-base, Aviation-BERT} × {γ=0, γ=1, γ=2, γ=5}.
- `pretrain_mlm.py` — domain-adaptive MLM pretraining on ASN narratives to build Aviation-BERT.
- `make_figures.py` — regenerates Fig 1 (class frequency, long-tail), Fig 4 (fine-tune curves), viz1/viz2 (per-class accuracy) from real run artifacts.
- `configs/` — YAML experiment configs.
- `notebooks/colab_runner.ipynb` — Colab-ready entry point.

## Running

Smoke-test on CPU with a 200-row subsample:

```bash
python train.py --config configs/smoke.yaml
```

Full run on Colab Pro A100:

```bash
python pretrain_mlm.py --config configs/aviation_bert_pretrain.yaml
python run_experiments.py --config configs/full_matrix.yaml
```

See `notebooks/colab_runner.ipynb` for the preferred path in Colab.
