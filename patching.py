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

    metric: Callable[[Tensor[N, seq, vocab]], Tensor[N]] — receives full logits [N, seq, vocab];
    callers select the position they need (e.g. lambda logits: metric_fn(logits[:, -1, :])).
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
    """For each candidate sender, patch to receiver_heads' Q/K/V inputs; measure metric change.

    Used for S2-inhibition (receiver_input="v" into NM heads) and
    induction analysis (receiver_input="k").

    metric: Callable[[Tensor[N, seq, vocab]], Tensor[N]] — receives full logits [N, seq, vocab];
    callers select the position they need (e.g. lambda logits: metric_fn(logits[:, -1, :])).
    """
    assert receiver_input in ("q", "k", "v")

    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_model  = model.config.n_embd
    d_head   = d_model // n_heads
    qkv_off  = {"q": 0, "k": d_model, "v": 2 * d_model}[receiver_input]
    recv_layers = sorted({l for l, _ in receiver_heads})

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

            # Patch sender head's output from corrupted run and measure effect
            with model.trace({"input_ids": clean_b}):
                model.transformer.h[sl].attn.c_proj.input[..., z_sl] = corr_z[sl][..., z_sl]
                patched_logits = model.lm_head.output.save()

            patched_m = metric(patched_logits.cpu())
            batch_effects[(sl, sh)].append((clean_m - patched_m).mean().item())
            del patched_logits
            clear_cache()

        del corr_z, clean_z, clean_logits
        clear_cache()

    scores = {k: sum(v) / len(v) for k, v in batch_effects.items()}
    return PatchingResult(scores=scores, n_layers=n_layers, n_heads=n_heads)
