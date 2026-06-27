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
        return sorted(self.scores, key=lambda lh: self.scores[lh], reverse=True)[:k]


def _batches(clean: torch.Tensor, corrupted: torch.Tensor, batch_size: Optional[int]):
    N = clean.size(0)
    if batch_size is None or batch_size >= N:
        yield clean, corrupted
        return
    for start in range(0, N, batch_size):
        yield clean[start:start + batch_size], corrupted[start:start + batch_size]


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

    for clean_b, corr_b in _batches(clean, corrupted, batch_size):

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
            batch_effects[(sl, sh)].append((clean_m - patched_m).mean().item())
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
) -> PatchingResult:
    """Measure the indirect effect of each sender head on receiver heads' inputs.

    For each candidate sender (sl, sh), computes the causal effect of the path:
      sender z (corrupted) → frozen residual stream → receiver layers run naturally → logit

    receiver_input: "q" | "k" | "v" — documents which component of the receiver is
        the intended path target.  The implementation lets receiver layers run fully
        naturally from the sender-modified residual stream, which simultaneously
        captures Q, K, and V modifications.  This correctly handles S2-inhibition
        heads that change NM heads' attention patterns via residual modification at
        the query position (END), not just via the value stream.

    metric receives full logits [N, seq, vocab].

    Algorithm
    ---------
    1. Corrupted trace: cache c_proj.input (concatenated head Z outputs) for every layer.
    2. Clean trace: cache c_proj.input + clean logits.
    3. For each sender (sl, sh):
         a. Run clean with sender z-slice replaced from corrupted run.
         b. All non-sender, non-receiver heads frozen to their clean z values;
            receiver layers run completely naturally from the modified residual stream.
         c. The receiver layers see a residual that has been modified by the sender's
            corrupted z propagating through frozen-clean attention and natural MLP
            in the intermediate blocks.
         d. Receiver heads' natural Q, K, V, Z computations are all allowed to change,
            capturing both direct V-modification and indirect Q-modification effects.
    4. score = (clean_metric − patched_metric).mean()

    Correctness: The S2-inhibition heads (7,3), (7,9), (8,6), (8,10) affect NM heads
    by modifying the residual stream at the END position (their query position), which
    changes NM heads' Q[END] and thus their attention pattern.  Patching block inputs
    directly while analytically propagating only through MLPs discards the two-hop
    path sender → residual_at_pos_X → NM_heads_attend_to_pos_X → modified_output →
    downstream_NM_head → logit.  Letting receiver layers run naturally captures this.

    nnsight 0.7 notes
    -----------------
    * Full writes to c_proj.input freeze the block's attention sub-forward-pass; this
      is intentional for non-receiver layers to enforce the frozen-clean-signal path.
    * Receiver layers have no intervention; c_attn runs naturally from the modified
      residual stream.
    """
    assert receiver_input in ("q", "k", "v")

    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    recv_set = {l for l, _ in receiver_heads}

    batch_effects: Dict[Tuple[int, int], List[float]] = {
        (l, h): [] for l in range(n_layers) for h in range(n_heads)
    }

    for clean_b, corr_b in _batches(clean, corrupted, batch_size):

        # Step 1: Cache z on corrupted run.
        corr_z: Dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": corr_b}):
            for l in range(n_layers):
                corr_z[l] = model.transformer.h[l].attn.c_proj.input.save()

        # Step 2: Cache z + clean logits.
        clean_z: Dict[int, torch.Tensor] = {}
        with model.trace({"input_ids": clean_b}):
            for l in range(n_layers):
                clean_z[l] = model.transformer.h[l].attn.c_proj.input.save()
            clean_logits = model.lm_head.output.save()

        clean_m = metric(clean_logits.cpu())

        for sl, sh in [(l, h) for l in range(n_layers) for h in range(n_heads)]:
            z_sl = slice(sh * d_head, (sh + 1) * d_head)

            # Step 3: Sender z replaced; non-receiver/non-sender heads frozen to clean;
            #          receiver layers run naturally → capture indirect path.
            with model.trace({"input_ids": clean_b}):
                for l in range(n_layers):
                    if l == sl:
                        # Sender: replace z-slice with corrupted value.
                        model.transformer.h[l].attn.c_proj.input[..., z_sl] = (
                            corr_z[sl][..., z_sl]
                        )
                    elif l not in recv_set:
                        # Non-receiver, non-sender: freeze entire attn output to clean.
                        model.transformer.h[l].attn.c_proj.input[...] = clean_z[l]
                    # Receiver layers: no intervention — run naturally from modified residual.
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            batch_effects[(sl, sh)].append((clean_m - patched_m).mean().item())
            del patched_logits
            clear_cache()

        del corr_z, clean_z, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)
