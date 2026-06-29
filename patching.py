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
        for (layer, head), v in self.scores.items():
            mat[layer, head] = v
        return mat

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        return sorted(self.scores, key=lambda lh: abs(self.scores[lh]), reverse=True)[
            :k
        ]


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
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads
    # head dimension, number of heads, number of layers

    batch_effects: Dict[Tuple[int, int], List[float]] = {
        (layer, head): [] for layer in range(n_layers) for head in range(n_heads)
    }

    for _, _, clean_b, corr_b in _batches(clean, corrupted, batch_size):
        corrupted_z = {}
        with model.trace({"input_ids": corr_b}):
            for layer in range(n_layers):
                corrupted_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

        clean_z = {}
        with model.trace({"input_ids": clean_b}):
            for layer in range(n_layers):
                clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()

        clean_m = metric(clean_logits.cpu())

        for sender_layer, sender_head in [
            (layer, head) for layer in range(n_layers) for head in range(n_heads)
        ]:
            sender_head_slice = slice(sender_head * d_head, (sender_head + 1) * d_head)
            with model.trace({"input_ids": clean_b}):
                for layer in range(n_layers):
                    if layer == sender_layer:
                        model.transformer.h[layer].attn.c_proj.input[
                            ..., sender_head_slice
                        ] = corrupted_z[sender_layer][..., sender_head_slice]
                    else:
                        model.transformer.h[layer].attn.c_proj.input[...] = clean_z[
                            layer
                        ]
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            # Sign matches paper (Wang et al. 2022): patched − clean is negative for helpful heads
            # (corrupting them hurts), positive for NNMs. Reversed from "contribution" framing.
            batch_effects[(sender_layer, sender_head)].append(
                (patched_m - clean_m).mean().item()
            )
            del patched_logits
            clear_cache()

        del corrupted_z, clean_z, clean_logits
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

    For each candidate sender (sender_layer, sender_head), measures the causal effect of the path:
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
    n_heads = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_heads
    # head dimension, number of heads, number of layers
    receiver_layers = {receiver_layer for receiver_layer, _ in receiver_heads}

    # Precompute QKV weights for receiver heads on CPU/float32 (avoids MPS memory limits).
    receiver_weights: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    for receiver_layer, receiver_head in receiver_heads:
        W = (
            model.transformer.h[receiver_layer]
            .attn.c_attn.weight.detach()
            .cpu()
            .float()
        )
        b = model.transformer.h[receiver_layer].attn.c_attn.bias.detach().cpu().float()
        head_slice_start, head_slice_end = (
            receiver_head * d_head,
            (receiver_head + 1) * d_head,
        )
        receiver_weights[(receiver_layer, receiver_head)] = {
            "W_Q": W[:, head_slice_start:head_slice_end],
            "b_Q": b[head_slice_start:head_slice_end],
            "W_K": W[:, d_model + head_slice_start : d_model + head_slice_end],
            "b_K": b[d_model + head_slice_start : d_model + head_slice_end],
            "W_V": W[:, 2 * d_model + head_slice_start : 2 * d_model + head_slice_end],
            "b_V": b[2 * d_model + head_slice_start : 2 * d_model + head_slice_end],
        }

    batch_effects: Dict[Tuple[int, int], List[float]] = {
        (layer, head): [] for layer in range(n_layers) for head in range(n_heads)
    }

    for batch_start, batch_end, clean_b, corr_b in _batches(
        clean, corrupted, batch_size
    ):
        sender_pos_b = (
            sender_positions[batch_start:batch_end].long()
            if sender_positions is not None
            else None
        )

        # Step 1: Cache z on corrupted.
        corrupted_z: Dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": corr_b}):
            for layer in range(n_layers):
                corrupted_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()

        # Step 2: Cache z + ln_1.output for receivers + logits.
        # All three receiver_input modes ("q", "k", "v") use ln_1.output as auxiliary:
        #   "q": patched Q, clean K/V  — ln1 used for Q; cln used for K/V
        #   "k": clean Q, patched K, clean V — ln1 used for K; cln used for Q/V
        #   "v": clean Q/K (for attn weights), patched V — ln1 used for V; cln for Q/K
        # nnsight 0.7: saving ln_1.output in the same trace as c_proj.input for all
        # layers triggers MissedProviderError when any post-receiver layer c_proj.input
        # is also saved (out-of-order proxy registration).  Split into two traces.
        clean_z: Dict[int, torch.Tensor] = {}
        clean_ln1: Dict[int, torch.Tensor] = {}

        with model.trace({"input_ids": clean_b}):
            for layer in range(n_layers):
                clean_z[layer] = model.transformer.h[layer].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()
        with model.trace({"input_ids": clean_b}):
            for layer in sorted(receiver_layers):
                # shape [N, seq, d_model] — input to c_attn after layer norm
                clean_ln1[layer] = model.transformer.h[layer].ln_1.output.save()

        clean_m = metric(clean_logits.cpu())

        for sender_layer, sender_head in [
            (layer, head) for layer in range(n_layers) for head in range(n_heads)
        ]:
            sender_head_slice = slice(sender_head * d_head, (sender_head + 1) * d_head)

            # Step 3: Patch sender, freeze non-receiver/non-sender; save receiver ln_1.
            # Receiver layers get NO c_proj.input write → c_attn runs naturally → ln_1 is valid.
            #
            # nnsight 0.7 constraint: writing c_proj.input for any layer AFTER the earliest
            # receiver prevents ln_1.output.save() from being provided for receiver layers.
            # Fix: only freeze layers STRICTLY BEFORE recv_min; post-receiver non-recv layers
            # are left to run freely (they do not affect receiver ln_1 inputs).
            # Saves must be registered in ascending layer order (nnsight proxy ordering).
            min_receiver_layer = min(receiver_layers)
            receiver_ln1: Dict[int, torch.Tensor] = {}
            with model.trace({"input_ids": clean_b}):
                for layer in range(min_receiver_layer):
                    if layer == sender_layer:
                        if sender_pos_b is not None:
                            # Position-restricted patch: only replace head sender_head at the
                            # specified token position per example, clean elsewhere.
                            repl = clean_z[layer].clone()
                            batch_idx = torch.arange(repl.shape[0])
                            repl[batch_idx, sender_pos_b, sender_head_slice] = (
                                corrupted_z[sender_layer][
                                    batch_idx, sender_pos_b, sender_head_slice
                                ]
                            )
                            model.transformer.h[layer].attn.c_proj.input[...] = repl
                        else:
                            model.transformer.h[layer].attn.c_proj.input[
                                ..., sender_head_slice
                            ] = corrupted_z[sender_layer][..., sender_head_slice]
                    elif layer not in receiver_layers:
                        model.transformer.h[layer].attn.c_proj.input[...] = clean_z[
                            layer
                        ]
                    # Receiver layers: no write → c_attn runs naturally from modified residual.
                for layer in sorted(
                    receiver_layers
                ):  # ascending order — nnsight requires in-order saves
                    if (
                        layer != sender_layer
                    ):  # guard: sender == receiver layer → c_attn skipped by slice-write
                        receiver_ln1[layer] = model.transformer.h[
                            layer
                        ].ln_1.output.save()

            # Analytically compute receiver z for step 4.
            # receiver_z[receiver_layer] starts as clean_z[receiver_layer]; receiver head slices are overwritten.
            receiver_z: Dict[int, torch.Tensor] = {}
            for receiver_layer, receiver_head in receiver_heads:
                if receiver_layer not in receiver_z:
                    receiver_z[receiver_layer] = clean_z[receiver_layer].clone()

                if receiver_layer == sender_layer:
                    # Same-layer: sender and receiver computed in parallel — no indirect path.
                    # Leave slice at clean value (receiver_z[receiver_layer] already initialised to clean_z).
                    continue

                w = receiver_weights[(receiver_layer, receiver_head)]
                ln1 = receiver_ln1[receiver_layer].cpu().float()  # [N, seq, d_model]

                cln = (
                    clean_ln1[receiver_layer].cpu().float()
                )  # [N, seq, d_model] — clean ln_1
                # Q, K, V projections for receiver head receiver_head — [N, seq, d_head]
                if receiver_input == "q":
                    Q_h = ln1 @ w["W_Q"] + w["b_Q"]  # patched
                    K_h = cln @ w["W_K"] + w["b_K"]  # clean
                    V_h = cln @ w["W_V"] + w["b_V"]  # clean
                elif receiver_input == "k":
                    Q_h = cln @ w["W_Q"] + w["b_Q"]  # clean
                    K_h = ln1 @ w["W_K"] + w["b_K"]  # patched
                    V_h = cln @ w["W_V"] + w["b_V"]  # clean
                else:  # "v": clean attention weights, patched V
                    Q_h = cln @ w["W_Q"] + w["b_Q"]  # clean
                    K_h = cln @ w["W_K"] + w["b_K"]  # clean
                    V_h = ln1 @ w["W_V"] + w["b_V"]  # patched
                seq = ln1.size(1)
                attn_scores = (Q_h @ K_h.transpose(-1, -2)) * (
                    d_head**-0.5
                )  # [N, seq, seq]
                causal_mask = torch.triu(
                    torch.ones(seq, seq, dtype=torch.bool), diagonal=1
                )
                attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
                z_h = torch.softmax(attn_scores, dim=-1) @ V_h  # [N, seq, d_head]

                # Move z_h to device/dtype of receiver_z before assignment.
                ref = receiver_z[receiver_layer]
                receiver_z[receiver_layer][
                    ..., receiver_head * d_head : (receiver_head + 1) * d_head
                ] = z_h.to(dtype=ref.dtype, device=ref.device)

            # Step 4: Replay receiver z into fresh clean run; freeze everything else.
            with model.trace({"input_ids": clean_b}):
                for layer in range(n_layers):
                    if layer in receiver_layers:
                        model.transformer.h[layer].attn.c_proj.input[...] = receiver_z[
                            layer
                        ]
                    else:
                        model.transformer.h[layer].attn.c_proj.input[...] = clean_z[
                            layer
                        ]
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            # Sign matches paper (Wang et al. 2022): patched − clean is negative for helpful heads.
            batch_effects[(sender_layer, sender_head)].append(
                (patched_m - clean_m).mean().item()
            )
            del patched_logits, receiver_ln1, receiver_z
            clear_cache()

        del corrupted_z, clean_z, clean_ln1, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)
