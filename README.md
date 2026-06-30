# Indirect Object Identification

Replication of the IOI circuit from [Wang et al. (2022)](https://arxiv.org/abs/2211.00593) using GPT-2 small and nnsight.

The circuit explains how GPT-2 identifies the indirect object in sentences like *"When Mary and John went to the store, John gave a drink to ___"* (answer: Mary). It comprises 26 attention heads across six functional types.

## Setup

```bash
pip install -r requirements.txt  # or: uv sync
```

## Running experiments

```bash
bash run_all_experiments.sh
```

Each experiment maps to a figure or appendix from the paper. They run independently and write plots to `plots/` and numeric results to `results/`.

## Layout

| File / folder | Purpose |
|---|---|
| `config.py` | Global constants: `SEED` (used by all experiments) |
| `ioi_dataset.py` | Dataset generation: IOI prompts, ABC counterfactuals, token indices |
| `patching.py` | Path patching: `path_patch_head_to_logits`, `path_patch_head_to_heads` |
| `circuit.py` | Circuit definition: `CIRCUIT`, `SEQ_POS_TO_KEEP`, `run_with_ablation` |
| `analysis.py` | Attention patterns, unembed projections, IO-direction projection, OV copy strength |
| `metrics.py` | `logit_diff`, `kl_divergence` — per-example scalar metrics |
| `ablation.py` | `Ablation` type + `mean_ablation`, `zero_ablation`, `resample_ablation`, `counterfactual_ablation` |
| `model.py` | `load_model` (GPT-2 small via nnsight) |
| `experiments/discovery/` | Head identification: name movers, SI, early heads |
| `experiments/validation/` | Circuit faithfulness, minimality, performance summary |
| `experiments/appendix/` | Appendix figures A, C–F, H–K, M |
| `tests/` | Regression tests for patching and dataset construction |

## Circuit heads (GPT-2 small)

| Type | Heads | Role |
|---|---|---|
| Name Mover (NM) | 9.9, 10.0, 9.6 | Write IO token to residual stream at END |
| Negative NM (NNM) | 10.7, 11.10 | Suppress the correct answer (negative copying) |
| Backup NM (BNM) | 10.10, 10.6, 10.2, 10.1, 11.2, 9.7, 9.0, 11.9 | Redundant copies, active when NM is ablated |
| S2-Inhibition (SI) | 7.3, 7.9, 8.6, 8.10 | Suppress S token at END via query path to NM |
| Induction (IH) | 5.5, 5.8, 5.9, 6.9 | Attend to S2 by matching S1→S2 repeat pattern |
| Duplicate Token (DT) | 0.1, 3.0 | Mark S1 position |
| Previous Token (PT) | 2.2, 4.11 | Carry positional signal one step forward |
