# indirect-object-identification/experiments/discovery/witness_pinning.py
"""Witness-pinning proof of concept on GPT-2 small IOI circuit.

Three claims under test:

  1. Witness-pinned ablation correctly scores primary necessity.
       Plain NIE underestimates primary necessity because backup heads compensate
       freely when the primary is ablated. Pinning the backup heads at their
       factual (clean) activations prevents compensation and reveals the primary's
       true necessity score.

  2. Witness importance scanning identifies the specific backup heads.
       For a fixed suspect (9,9), scanning candidate witnesses one at a time
       and measuring the importance delta should rank the backup name movers
       near the top — without any prior knowledge of the circuit.

  3. Denoising-based ablation (Dn / PIE) fails to find preemption backup heads.
       Dn recovers OR-gate participants — heads sufficient on their own — by
       restoring clean activations in a corrupted context. Prior work validated
       circuit completeness in aggregate on the IOI task but never checked
       whether Dn specifically recovers Wang et al.'s backup name movers.
       Mueller (2024) classifies backup name movers as preemption rather than
       overdetermination: they are dormant in the clean pass and activate only
       when the primary is absent. If Mueller is right, their clean activation
       carries no task signal, so Dn returns near-zero scores for them.
       Witness pinning finds them regardless, because it prevents compensation
       rather than relying on the backup's clean activation being informative.

Expected results under Mueller's preemption hypothesis:
  - NIE: primaries score near-zero (backup compensates freely).
  - Witness-pinned: primaries score significantly more negative (necessity revealed).
  - PIE/Dn: backup name movers score near-zero (dormant clean activations).
  - Witness importance: backup name movers rank high (compensation prevented).

Expected results if Wang et al. overdetermination hypothesis is correct instead:
  - PIE/Dn: backup name movers score positively (active clean activations).
  - Witness importance: backup name movers still rank high.
  - Either way, witness-pinned primaries score more negative than NIE.

The PIE recall vs. witness importance recall table is the key empirical test
between the two hypotheses.
"""

import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from circuit import CIRCUIT
from config import SEED
from config import N as N_EXAMPLES
from ioi_dataset import IOIDataset
from metrics import logit_diff
from model import load_model
from witness import (
    pie_denoising_scores,
    witness_importance_scores,
    witness_pinned_ablation_scores,
)

PRIMARY_NAME_MOVERS = set(CIRCUIT["name_mover"])  # (9,9),(10,0),(9,6)
BACKUP_NAME_MOVERS = set(CIRCUIT["backup_name_mover"])
# Non-circuit heads used as negative controls: Pinned≈NIE expected, importance≈0.
CONTROL_HEADS = [(2, 0), (11, 0)]


def _build_dataset(tokenizer):
    random.seed(SEED)
    np.random.seed(SEED)
    ioi = IOIDataset("mixed", N=N_EXAMPLES, tokenizer=tokenizer, prepend_bos=False)
    # ABC baseline: randomise IO, then both S positions — identical to fig3.
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    abc = abc.gen_flipped_prompts(("S1", "RAND"))
    return ioi, abc


def _end_metric(ioi, n):
    end_pos = ioi.word_idx["end"].long()

    def metric(logits):
        return logit_diff(
            logits[torch.arange(n), end_pos],
            ioi.io_tokenIDs,
            ioi.s_tokenIDs,
        )

    return metric


def _load_path_patch_scores():
    """Load pre-computed path-patching scores from the fig3 experiment results."""
    path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "results",
        "name_movers",
        "head_to_logits_causal_effect.csv",
    )
    scores = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                scores[(int(row["layer"]), int(row["head"]))] = float(
                    row["causal_effect"]
                )
    return scores


def _print_head_table(title, nie_scores, pinned_scores, pie_scores, path_scores):
    heads_of_interest = (
        list(PRIMARY_NAME_MOVERS) + sorted(BACKUP_NAME_MOVERS) + list(CONTROL_HEADS)
    )
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("  (NIE/Pinned/PathPatch: neg=helpful; PIE: pos=helpful)")
    print(f"{'=' * 70}")
    print(
        f"  {'Head':<10} {'NIE':>10} {'Pinned':>10} {'PIE':>10} {'PathPatch':>10}  Role"
    )
    print(f"  {'-' * 64}")
    for head in heads_of_interest:
        if head in PRIMARY_NAME_MOVERS:
            role = "primary"
        elif head in BACKUP_NAME_MOVERS:
            role = "backup"
        else:
            role = "control"
        nie = nie_scores.get(head, float("nan"))
        pinned = pinned_scores.get(head, float("nan"))
        pie = pie_scores.get(head, float("nan"))
        path = path_scores.get(head, float("nan"))
        print(
            f"  {str(head):<10} {nie:>10.4f} {pinned:>10.4f}"
            f" {pie:>10.4f} {path:>10.4f}  {role}"
        )


def _print_importance_table(suspect, importance, k_values=(4, 8, 12)):
    ranked = sorted(importance, key=lambda h: importance[h], reverse=True)
    print(f"\n{'=' * 70}")
    print(f"  Witness importance ranking for suspect {suspect}")
    print("  (importance = |pinned_score| - |baseline_score|, higher is more)")
    print(f"{'=' * 70}")
    print(f"  {'Rank':<6} {'Head':<10} {'Importance':>12}  Backup?")
    print(f"  {'-' * 40}")
    for rank, head in enumerate(ranked[:15], 1):
        is_backup = "YES" if head in BACKUP_NAME_MOVERS else "no"
        print(f"  {rank:<6} {str(head):<10} {importance[head]:>12.4f}  {is_backup}")

    print()
    for k in k_values:
        top_k = set(ranked[:k])
        recall = len(top_k & BACKUP_NAME_MOVERS) / len(BACKUP_NAME_MOVERS)
        print(f"  Recall of backup name movers in top-{k}: {recall:.2f}")


def _print_dn_crosscheck(pie_scores, importance, k_values=(4, 8, 12)):
    """Compare PIE/Dn recall vs. witness importance recall on backup name movers.

    This is the key empirical test of the Mueller/Wang et al. disagreement:
      - High PIE recall → backups are overdetermination (active clean activations).
      - Near-zero PIE recall → backups are preemption (dormant clean activations),
        and Dn fails on precisely the heads Wang et al. found.
    Witness importance recall is shown alongside for direct comparison.

    Scope note: PIE ranks heads globally (all 144); importance ranks all 144 from
    the suspect's perspective — comparable only if backup heads rank high regardless
    of which primary is chosen.
    """
    # PIE: positive scores mean the head's clean activation restores metric in
    # corrupted context. Higher = more task signal in clean pass.
    pie_ranked = sorted(pie_scores, key=lambda h: pie_scores[h], reverse=True)
    importance_ranked = sorted(importance, key=lambda h: importance[h], reverse=True)

    print(f"\n{'=' * 70}")
    print("  Dn cross-check: do PIE scores recover Wang et al.'s backup heads?")
    print("  High PIE recall → overdetermination. Low recall → preemption (Mueller).")
    print(f"{'=' * 70}")
    print(f"  {'k':<6} {'PIE recall':>12} {'Witness recall':>16}")
    print(f"  {'-' * 36}")
    for k in k_values:
        pie_top_k = set(pie_ranked[:k])
        imp_top_k = set(importance_ranked[:k])
        pie_recall = len(pie_top_k & BACKUP_NAME_MOVERS) / len(BACKUP_NAME_MOVERS)
        imp_recall = len(imp_top_k & BACKUP_NAME_MOVERS) / len(BACKUP_NAME_MOVERS)
        print(f"  {k:<6} {pie_recall:>12.2f} {imp_recall:>16.2f}")

    print(f"\n{'=' * 70}")
    print("  Per-head classification: PIE score vs. witness importance")
    print(
        "  PIE≈0 + high importance → preemption. PIE>0 + high importance → overdetermination."
    )
    print(f"{'=' * 70}")
    print(f"  {'Head':<10} {'PIE score':>12} {'Importance':>12}  Classification")
    print(f"  {'-' * 52}")
    for head in sorted(BACKUP_NAME_MOVERS):
        pie = pie_scores.get(head, float("nan"))
        imp = importance.get(head, float("nan"))
        # Threshold: PIE > 0.05 logit diff units indicates informative clean activation.
        # This is heuristic — inspect the full distribution to calibrate.
        classification = "overdetermination" if pie > 0.05 else "preemption"
        print(f"  {str(head):<10} {pie:>12.4f} {imp:>12.4f}  {classification}")


def run():
    torch.set_grad_enabled(False)
    model = load_model()
    ioi, abc = _build_dataset(model.tokenizer)
    N = len(ioi)
    clean_toks = ioi.toks.long()
    corrupted_toks = abc.toks.long()
    metric = _end_metric(ioi, N)
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    end_pos = ioi.word_idx["end"].long()

    # --- Comparison 1: head-by-head scores under all four estimands ----------

    print("Running NIE ablation (no witnesses)...")
    nie_result = witness_pinned_ablation_scores(
        model,
        clean_toks,
        corrupted_toks,
        metric,
        witness_heads=[],
        positions=end_pos,
    )

    print("Running witness-pinned ablation (witnesses = backup name movers)...")
    pinned_result = witness_pinned_ablation_scores(
        model,
        clean_toks,
        corrupted_toks,
        metric,
        witness_heads=list(BACKUP_NAME_MOVERS),
        positions=end_pos,
    )

    print("Running PIE denoising...")
    pie_result = pie_denoising_scores(
        model,
        clean_toks,
        corrupted_toks,
        metric,
    )

    path_scores = _load_path_patch_scores()

    _print_head_table(
        "Necessity scores — primary and backup name movers",
        nie_result.scores,
        pinned_result.scores,
        pie_result.scores,
        path_scores,
    )

    # --- (9,6) all-positions diagnostic ---
    # (9,6) showed a sign-inverted NIE when ablated only at end_pos.
    # Run with positions=None (all positions ablated) to check if the sign flips.
    print("\nRunning (9,6) all-positions NIE (sign-inversion diagnostic)...")
    nie_full_96 = witness_pinned_ablation_scores(
        model,
        clean_toks,
        corrupted_toks,
        metric,
        witness_heads=[],
        positions=None,
        suspect_heads=[(9, 6)],
    )
    nie_end_96 = nie_result.scores.get((9, 6), float("nan"))
    nie_all_96 = nie_full_96.scores[(9, 6)]
    pinned_end_96 = pinned_result.scores.get((9, 6), float("nan"))
    print("\n  (9,6) position ablation comparison:")
    print(f"    NIE end_pos only:  {nie_end_96:+.4f}")
    print(f"    NIE all positions: {nie_all_96:+.4f}")
    print(f"    Pinned end_pos:    {pinned_end_96:+.4f}")

    # --- Comparison 2: witness importance for all three primaries + control ---

    all_heads = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
    importance_by_suspect = {}
    for suspect in list(PRIMARY_NAME_MOVERS) + [(11, 0)]:
        role = "primary" if suspect in PRIMARY_NAME_MOVERS else "negative control"
        print(f"\nRunning witness importance scan for suspect {suspect} ({role})...")
        importance_by_suspect[suspect] = witness_importance_scores(
            model,
            clean_toks,
            corrupted_toks,
            metric,
            suspect_head=suspect,
            candidate_witnesses=all_heads,
            positions=end_pos,
        )
        _print_importance_table(suspect, importance_by_suspect[suspect])

    # --- Comparison 3 & 4: Dn cross-check and preemption classification ---
    # Use (9,9) as representative primary — strongest NIE→Pinned signal.
    _print_dn_crosscheck(pie_result.scores, importance_by_suspect[(9, 9)])


if __name__ == "__main__":
    run()
