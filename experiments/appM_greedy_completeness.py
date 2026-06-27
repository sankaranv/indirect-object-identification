r"""Appendix M / Figs 6, 22: Greedy completeness K sets and scatter plots.

Fig 6a scatter: (F(C\K), F(M\K)) for the 26-head circuit — points near the
diagonal show the circuit is complete (non-circuit heads contribute ~0).

Fig 6b scatter: same axes for the naive circuit (NM+SI only) — points well
below the diagonal show the naive circuit is incomplete.

Fig 22 / results CSV: K sets found by greedy Algorithm 3.
"""
import os, sys, random, csv, torch
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data", "ioi"))
torch.set_grad_enabled(False)

from utils import load_model
from metrics import logit_diff
from circuit import CIRCUIT, SEQ_POS_TO_KEEP, compute_means, run_with_mean_ablation
from ioi_dataset import IOIDataset


NAIVE_CIRCUIT = {
    "name_mover":          CIRCUIT["name_mover"],
    "s2_inhibition":       CIRCUIT["s2_inhibition"],
    "backup_name_mover":   [],
    "negative_name_mover": [],
    "induction":           [],
    "duplicate_token":     [],
    "previous_token":      [],
}


def _ablate_circuit(base_circuit, heads_to_remove):
    """Return a copy of base_circuit with heads_to_remove excluded."""
    return {t: [lh for lh in heads if lh not in heads_to_remove]
            for t, heads in base_circuit.items()}


def _ld(logits, N, end_pos, io_ids, s_ids):
    return logit_diff(
        logits.cpu()[torch.arange(N), end_pos],
        io_ids, s_ids,
    ).mean().item()


def _F(model, ioi, means, circuit, word_idx):
    """Circuit performance: run model with circuit mean-ablation, return LD."""
    logits = run_with_mean_ablation(
        model, ioi.toks.long(), means, circuit, SEQ_POS_TO_KEEP, word_idx
    )
    N, end_pos = len(ioi), ioi.word_idx["end"]
    return _ld(logits, N, end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs)


def _F_model_ablate_K(model, ioi, means, K):
    """Full model logit diff with only K heads mean-ablated (no circuit restriction)."""
    n_layers = len(model.transformer.h)
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    N, seq = ioi.toks.shape
    # Build per-layer keep masks (True = keep clean, False = replace with mean)
    keep = {l: torch.ones(N, seq, n_heads, dtype=torch.bool) for l in range(n_layers)}
    for (l, h) in K:
        keep[l][:, :, h] = False
    layers_to_patch = [l for l in range(n_layers) if not keep[l].all()]
    end_pos = ioi.word_idx["end"]
    with model.trace({"input_ids": ioi.toks.long()}):
        for l in layers_to_patch:
            z = model.transformer.h[l].attn.c_proj.input
            z_h = z.reshape(N, seq, n_heads, d_head)
            mask = keep[l].unsqueeze(-1).to(z.device)
            z_new = torch.where(mask, z_h, means[l].to(z.device))
            model.transformer.h[l].attn.c_proj.input[:] = z_new.reshape(N, seq, n_heads * d_head)
        logits = model.lm_head.output.save()
    return _ld(logits, N, end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs)


def greedy_k_sample(model, ioi, means, base_circuit, k=10, n_steps=5, seed=0):
    """Algorithm 3: greedy K sampling.

    At each step, sample k heads from (circuit \\ K), find which removal most
    reduces circuit performance, add it to K.  Returns K as a set of (l,h).
    """
    rng = random.Random(seed)
    circuit_heads = [lh for heads in base_circuit.values() for lh in heads]
    K = set()
    for _ in range(n_steps):
        available = [lh for lh in circuit_heads if lh not in K]
        if not available:
            break
        sample = rng.sample(available, min(k, len(available)))
        # baseline: F(C \ K)
        f_k = _F(model, ioi, means, _ablate_circuit(base_circuit, K), ioi.word_idx)
        best_v, best_delta = None, -1.0
        for v in sample:
            f_kv = _F(model, ioi, means, _ablate_circuit(base_circuit, K | {v}), ioi.word_idx)
            delta = abs(f_kv - f_k)
            if delta > best_delta:
                best_delta, best_v = delta, v
        if best_v is not None:
            K.add(best_v)
    return K


def run():
    model = load_model()
    random.seed(1)
    np.random.seed(1)

    ioi = IOIDataset("mixed", N=100, tokenizer=model.tokenizer, prepend_bos=False)
    abc = ioi.gen_flipped_prompts(("IO", "RAND"))
    abc = abc.gen_flipped_prompts(("S", "RAND"))
    N = len(ioi)
    end_pos = ioi.word_idx["end"]

    print("Computing ABC means…")
    means = compute_means(model, abc.toks.long(), abc.groups)

    with model.trace({"input_ids": ioi.toks.long()}):
        full_logits = model.lm_head.output.save()
    full_ld = _ld(full_logits, N, end_pos, ioi.io_tokenIDs, ioi.s_tokenIDs)
    print(f"Full model LD: {full_ld:.4f}")

    # ------------------------------------------------------------------
    # Fig 22: Greedy K sets
    # ------------------------------------------------------------------
    print("Running greedy K sampling (10 seeds, n_steps=5, k=10)…")
    greedy_runs = []
    for seed in range(10):
        K = greedy_k_sample(model, ioi, means, CIRCUIT, k=10, n_steps=5, seed=seed)
        # F(C\K): circuit minus K
        f_ck = _F(model, ioi, means, _ablate_circuit(CIRCUIT, K), ioi.word_idx)
        # F(M\K): full model with only K ablated (no circuit restriction)
        f_mk = _F_model_ablate_K(model, ioi, means, K)
        incompleteness = abs(f_ck - f_mk)
        greedy_runs.append((K, f_ck, f_mk, incompleteness))
        print(f"  seed={seed}: |K|={len(K)}, F(C\\K)={f_ck:.3f}, incompleteness={incompleteness:.4f}")

    greedy_runs.sort(key=lambda x: x[3], reverse=True)
    top5 = greedy_runs[:5]

    os.makedirs("results/circuit", exist_ok=True)
    with open("results/circuit/fig22_greedy_k_sets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rank", "K_set", "K_size", "F_circuit_minus_K", "F_model_minus_K", "incompleteness"
        ])
        w.writeheader()
        for i, (K, fck, fmk, inc) in enumerate(top5):
            w.writerow({
                "rank": i + 1,
                "K_set": str(sorted(K)),
                "K_size": len(K),
                "F_circuit_minus_K": round(fck, 4),
                "F_model_minus_K":   round(fmk, 4),
                "incompleteness":    round(inc, 6),
            })
    print("Saved results/circuit/fig22_greedy_k_sets.csv")

    # ------------------------------------------------------------------
    # Fig 6a: scatter for complete 26-head circuit
    # For complete circuit: F(C\K) ≈ F(M\K) → points near diagonal.
    # ------------------------------------------------------------------
    print("Generating random K sets for Fig 6a scatter (complete circuit)…")
    all_heads = [lh for heads in CIRCUIT.values() for lh in heads]
    scatter_circuit = []
    rng = random.Random(42)
    for size in range(1, min(len(all_heads), 15)):
        for _ in range(3):
            K = set(rng.sample(all_heads, size))
            fck = _F(model, ioi, means, _ablate_circuit(CIRCUIT, K), ioi.word_idx)
            fmk = _F_model_ablate_K(model, ioi, means, K)
            scatter_circuit.append((fck, fmk))

    # Add greedy K sets to scatter
    for K, fck, fmk, _ in greedy_runs:
        scatter_circuit.append((fck, fmk))

    # ------------------------------------------------------------------
    # Fig 6b: scatter for naive circuit (NM+SI only)
    # F(C\K): naive circuit performance; F(M\K): complete circuit performance.
    # Points off diagonal → naive circuit is incomplete.
    # ------------------------------------------------------------------
    print("Generating random K sets for Fig 6b scatter (naive circuit)…")
    naive_heads = [lh for heads in NAIVE_CIRCUIT.values() for lh in heads]
    scatter_naive = []
    for size in range(1, min(len(naive_heads) + 1, 9)):
        for _ in range(3):
            if len(naive_heads) < size:
                K = set(naive_heads)
            else:
                K = set(rng.sample(naive_heads, size))
            # Circuit performance: naive circuit minus K
            fck = _F(model, ioi, means, _ablate_circuit(NAIVE_CIRCUIT, K), ioi.word_idx)
            # Model performance proxy: complete circuit minus K
            fmk = _F(model, ioi, means, _ablate_circuit(CIRCUIT, K), ioi.word_idx)
            scatter_naive.append((fck, fmk))

    # ------------------------------------------------------------------
    # Save scatter plots
    # ------------------------------------------------------------------
    os.makedirs("plots/circuit", exist_ok=True)
    for scatter, fname, title in [
        (scatter_circuit, "fig6a_scatter.png", "Completeness scatter: 26-head circuit"),
        (scatter_naive,   "fig6b_scatter.png", "Completeness scatter: naive circuit (NM+SI only)"),
    ]:
        if not scatter:
            continue
        xs, ys = zip(*scatter)
        lim_lo = min(min(xs), min(ys)) - 0.2
        lim_hi = max(max(xs), max(ys)) + 0.2
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xs, ys, alpha=0.7, s=25, color="#4C72B0")
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, label="y = x")
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("F(C \\ K): circuit logit diff after removing K")
        ax.set_ylabel("F(M \\ K): model logit diff after removing K")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"plots/circuit/{fname}", dpi=150)
        plt.close()
        print(f"Saved plots/circuit/{fname}")

    print(f"\nSummary:")
    print(f"  Complete-circuit scatter points: {len(scatter_circuit)}")
    print(f"  Naive-circuit scatter points:    {len(scatter_naive)}")
    print(f"  Greedy K sets (top 5 by incompleteness): {len(top5)}")


if __name__ == "__main__":
    run()
