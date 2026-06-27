from __future__ import annotations

import dataclasses
from typing import Callable, Dict, List, Optional, Tuple

import torch

from utils import clear_cache


@dataclasses.dataclass
class PatchingResult:
    """Causal effect score for each (layer, head) pair."""
    scores: Dict[Tuple[int, int], float]
    n_layers: int
    n_heads: int

    def as_matrix(self) -> torch.Tensor:
        mat = torch.zeros(self.n_layers, self.n_heads)
        for (l, h), v in self.scores.items():
            mat[l, h] = v
        return mat

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        return sorted(self.scores, key=lambda lh: abs(self.scores[lh]), reverse=True)[:k]


def _batches(clean: torch.Tensor, corrupted: torch.Tensor, batch_size: Optional[int]):
    N = clean.size(0)
    if batch_size is None or batch_size >= N:
        yield 0, N, clean, corrupted
        return
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        yield start, end, clean[start:end], corrupted[start:end]


def path_patch_head_to_logits(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: Optional[int] = None,
) -> PatchingResult:
    """For each head, patch its z-output from corrupted into a clean run; measure metric change.

    metric: Callable[[Tensor[N, seq, vocab]], Tensor[N]] — receives full logits [N, seq, vocab].
    """
    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads

    batch_effects: Dict[Tuple[int, int], List[float]] = {
        (l, h): [] for l in range(n_layers) for h in range(n_heads)
    }

    for _, _, clean_b, corr_b in _batches(clean, corrupted, batch_size):

        corr_z = {}
        with model.trace({"input_ids": corr_b}):
            for l in range(n_layers):
                corr_z[l] = model.transformer.h[l].attn.c_proj.input.save()

        clean_z = {}
        with model.trace({"input_ids": clean_b}):
            for l in range(n_layers):
                clean_z[l] = model.transformer.h[l].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()

        clean_m = metric(clean_logits.cpu())

        for sl, sh in [(l, h) for l in range(n_layers) for h in range(n_heads)]:
            z_sl = slice(sh * d_head, (sh + 1) * d_head)
            with model.trace({"input_ids": clean_b}):
                for l in range(n_layers):
                    if l == sl:
                        model.transformer.h[l].attn.c_proj.input[..., z_sl] = corr_z[sl][..., z_sl]
                    else:
                        model.transformer.h[l].attn.c_proj.input[...] = clean_z[l]
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            # Sign matches paper (Wang et al. 2022): patched − clean is negative for helpful heads
            # (corrupting them hurts), positive for NNMs. Reversed from "contribution" framing.
            batch_effects[(sl, sh)].append((patched_m - clean_m).mean().item())
            del patched_logits
            clear_cache()

        del corr_z, clean_z, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)


def path_patch_head_to_heads(
    model,
    clean: torch.Tensor,
    corrupted: torch.Tensor,
    receiver_heads: List[Tuple[int, int]],
    receiver_input: str,
    metric: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: Optional[int] = None,
    sender_positions: Optional[torch.Tensor] = None,
) -> PatchingResult:
    """Measure the indirect effect of each sender head on receiver heads' Q/K/V inputs.

    For each candidate sender (sl, sh), measures the causal effect of the path:
      sender z (corrupted) → residual stream → receiver heads' Q/K/V → logit

    receiver_input selects which component of the receiver is isolated:
      "v" — value input (S2-inhibition → NM): uses clean attn weights × patched V
      "q" — query input: recomputes attn with patched Q, clean K and V
      "k" — key input: recomputes attn with clean Q, patched K, clean V

    sender_positions: optional LongTensor [N] restricting the patch to one token
      position per example (e.g. word_idx["S2"] for Fig 12b). When None, the
      sender's full z is patched at all positions (original behaviour).

    metric receives full logits [N, seq, vocab]; caller selects the position.

    Algorithm (4-step indirect-effect):
    1. Corrupted trace: cache z (c_proj.input) for every layer.
    2. Clean trace: cache z + receiver attn weights ("v") or receiver ln_1 ("q"/"k") + logits.
    3. Per sender: run clean with sender z patched; freeze non-receiver/non-sender layers;
       save receiver ln_1.output (no c_proj.input write for receivers → c_attn runs naturally).
    4. Analytically compute receiver head z from saved ln_1 + weights; replay into clean run.
       Effect = clean_metric − patched_metric.
    """
    assert receiver_input in ("q", "k", "v")

    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_model  = model.config.n_embd
    d_head   = d_model // n_heads
    recv_set = {l for l, _ in receiver_heads}

    # Precompute QKV weights for receiver heads on CPU/float32 (avoids MPS memory limits).
    recv_W: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for rl, rh in receiver_heads:
        W = model.transformer.h[rl].attn.c_attn.weight.detach().cpu().float()
        b = model.transformer.h[rl].attn.c_attn.bias.detach().cpu().float()
        qs, qe = rh * d_head, (rh + 1) * d_head
        recv_W[(rl, rh)] = {
            "W_Q": W[:, qs:qe],            "b_Q": b[qs:qe],
            "W_K": W[:, d_model + qs : d_model + qe],
            "b_K": b[d_model + qs : d_model + qe],
            "W_V": W[:, 2 * d_model + qs : 2 * d_model + qe],
            "b_V": b[2 * d_model + qs : 2 * d_model + qe],
        }

    batch_effects: Dict[Tuple[int, int], List[float]] = {
        (l, h): [] for l in range(n_layers) for h in range(n_heads)
    }

    for batch_start, batch_end, clean_b, corr_b in _batches(clean, corrupted, batch_size):
        sender_pos_b = (
            sender_positions[batch_start:batch_end].long()
            if sender_positions is not None else None
        )

        # Step 1: Cache z on corrupted.
        corr_z: Dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": corr_b}):
            for l in range(n_layers):
                corr_z[l] = model.transformer.h[l].attn.c_proj.input.save()

        # Step 2: Cache z + auxiliary (attn weights for "v"; ln_1 for "q"/"k") + logits.
        clean_z: Dict[int, torch.Tensor] = {}
        clean_aux: Dict[int, torch.Tensor] = {}

        if receiver_input == "v":
            with model.trace({"input_ids": clean_b}, output_attentions=True):
                for l in range(n_layers):
                    clean_z[l] = model.transformer.h[l].attn.c_proj.input.save()
                for l in recv_set:
                    # shape [N, n_heads, seq, seq] — attention probabilities
                    clean_aux[l] = model.transformer.h[l].attn.output[1].save()
                clean_logits = model.lm_head.output.save()
        else:
            # nnsight 0.7: saving ln_1.output in the same trace as c_proj.input for all
            # layers triggers MissedProviderError when any post-receiver layer c_proj.input
            # is also saved (out-of-order proxy registration).  Split into two traces.
            with model.trace({"input_ids": clean_b}):
                for l in range(n_layers):
                    clean_z[l] = model.transformer.h[l].attn.c_proj.input.save()
                clean_logits = model.lm_head.output.save()
            with model.trace({"input_ids": clean_b}):
                for l in sorted(recv_set):
                    # shape [N, seq, d_model] — input to c_attn after layer norm
                    clean_aux[l] = model.transformer.h[l].ln_1.output.save()

        clean_m = metric(clean_logits.cpu())

        for sl, sh in [(l, h) for l in range(n_layers) for h in range(n_heads)]:
            z_sl = slice(sh * d_head, (sh + 1) * d_head)

            # Step 3: Patch sender, freeze non-receiver/non-sender; save receiver ln_1.
            # Receiver layers get NO c_proj.input write → c_attn runs naturally → ln_1 is valid.
            #
            # nnsight 0.7 constraint: writing c_proj.input for any layer AFTER the earliest
            # receiver prevents ln_1.output.save() from being provided for receiver layers.
            # Fix: only freeze layers STRICTLY BEFORE recv_min; post-receiver non-recv layers
            # are left to run freely (they do not affect receiver ln_1 inputs).
            # Saves must be registered in ascending layer order (nnsight proxy ordering).
            recv_min_l = min(recv_set)
            recv_ln1: Dict[int, torch.Tensor] = {}
            with model.trace({"input_ids": clean_b}):
                for l in range(recv_min_l):
                    if l == sl:
                        if sender_pos_b is not None:
                            # Position-restricted patch: only replace head sh at the
                            # specified token position per example, clean elsewhere.
                            repl = clean_z[l].clone()
                            idx_b = torch.arange(repl.shape[0])
                            repl[idx_b, sender_pos_b, z_sl] = corr_z[sl][idx_b, sender_pos_b, z_sl]
                            model.transformer.h[l].attn.c_proj.input[...] = repl
                        else:
                            model.transformer.h[l].attn.c_proj.input[..., z_sl] = corr_z[sl][..., z_sl]
                    elif l not in recv_set:
                        model.transformer.h[l].attn.c_proj.input[...] = clean_z[l]
                    # Receiver layers: no write → c_attn runs naturally from modified residual.
                for l in sorted(recv_set):  # ascending order — nnsight requires in-order saves
                    if l != sl:  # guard: sender == receiver layer → c_attn skipped by slice-write
                        recv_ln1[l] = model.transformer.h[l].ln_1.output.save()

            # Analytically compute receiver z for step 4.
            # recv_z_step4[rl] starts as clean_z[rl]; receiver head slices are overwritten.
            recv_z_step4: Dict[int, torch.Tensor] = {}
            for rl, rh in receiver_heads:
                if rl not in recv_z_step4:
                    recv_z_step4[rl] = clean_z[rl].clone()

                if rl == sl:
                    # Same-layer: sender and receiver computed in parallel — no indirect path.
                    # Leave slice at clean value (recv_z_step4[rl] already initialised to clean_z).
                    continue

                w   = recv_W[(rl, rh)]
                ln1 = recv_ln1[rl].cpu().float()  # [N, seq, d_model]

                if receiver_input == "v":
                    V_h = ln1 @ w["W_V"] + w["b_V"]                             # [N, seq, d_head]
                    attn_h = clean_aux[rl].cpu().float()[:, rh, :, :]           # [N, seq, seq]
                    z_h = attn_h @ V_h                                           # [N, seq, d_head]
                else:
                    cln = clean_aux[rl].cpu().float()                            # [N, seq, d_model]
                    if receiver_input == "q":
                        Q_h = ln1 @ w["W_Q"] + w["b_Q"]
                        K_h = cln @ w["W_K"] + w["b_K"]
                        V_h = cln @ w["W_V"] + w["b_V"]
                    else:  # "k"
                        Q_h = cln @ w["W_Q"] + w["b_Q"]
                        K_h = ln1 @ w["W_K"] + w["b_K"]
                        V_h = cln @ w["W_V"] + w["b_V"]
                    seq = ln1.size(1)
                    scores = (Q_h @ K_h.transpose(-1, -2)) * (d_head ** -0.5)  # [N, seq, seq]
                    causal_mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
                    scores = scores.masked_fill(causal_mask, float("-inf"))
                    z_h = torch.softmax(scores, dim=-1) @ V_h                   # [N, seq, d_head]

                # Move z_h to device/dtype of recv_z_step4 before assignment.
                ref = recv_z_step4[rl]
                recv_z_step4[rl][..., rh * d_head : (rh + 1) * d_head] = (
                    z_h.to(dtype=ref.dtype, device=ref.device)
                )

            # Step 4: Replay receiver z into fresh clean run; freeze everything else.
            with model.trace({"input_ids": clean_b}):
                for l in range(n_layers):
                    if l in recv_set:
                        model.transformer.h[l].attn.c_proj.input[...] = recv_z_step4[l]
                    else:
                        model.transformer.h[l].attn.c_proj.input[...] = clean_z[l]
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            # Sign matches paper (Wang et al. 2022): patched − clean is negative for helpful heads.
            batch_effects[(sl, sh)].append((patched_m - clean_m).mean().item())
            del patched_logits, recv_ln1, recv_z_step4
            clear_cache()

        del corr_z, clean_z, clean_aux, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)
