from __future__ import annotations

from typing import Dict, Tuple

import torch

from utils import clear_cache


def attention_to_positions(
    model,
    tokens: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> Dict[Tuple[int, int], float]:
    """Mean attention probability from query_positions to key_positions for each head.

    tokens           : [N, seq]
    query_positions  : [N] — per-example query token index
    key_positions    : [N] — per-example key token index
    Returns          : {(layer, head): mean_attn_prob}
    """
    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    N        = tokens.size(0)

    result: Dict[Tuple[int, int], float] = {}
    for layer in range(n_layers):
        with model.trace({"input_ids": tokens}, output_attentions=True):
            w = model.transformer.h[layer].attn.output[1].save()
        # w: [N, n_heads, seq, seq]
        for head in range(n_heads):
            probs = w[torch.arange(N), head, query_positions, key_positions]
            result[(layer, head)] = probs.mean().item()
        del w
        clear_cache()

    return result


def unembed_projections_at_positions(
    model,
    tokens: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Project each layer's residual stream at positions through the unembedding matrix.

    Returns pseudo-logits [n_layers, N, vocab] — not real model outputs; a view through
    W_U of what each layer's residual state "represents" in vocabulary space.

    tokens    : [N, seq]
    positions : [N] — per-example token index to read from
    """
    n_layers = len(model.transformer.h)
    N        = tokens.size(0)
    vocab    = model.config.vocab_size

    W_U  = model.lm_head.weight.detach()   # [vocab, d_model]
    out  = torch.zeros(n_layers, N, vocab)

    for layer in range(n_layers):
        with model.trace({"input_ids": tokens}):
            resid = model.transformer.h[layer].output.save()
        # resid: [N, seq, d_model]  — nnsight 0.7 unwraps the block output tuple
        resid_at_pos = resid[torch.arange(N), positions, :]   # [N, d_model]
        out[layer]   = resid_at_pos @ W_U.T
        del resid
        clear_cache()

    return out


def head_output_io_projection(
    model,
    tokens: torch.Tensor,
    end_positions: torch.Tensor,
    io_token_ids: list,
    s_token_ids: list,
) -> Dict[Tuple[int, int], torch.Tensor]:
    """Per-example projection of each head's output at END onto the IO−S direction.

    For each head, computes: (z_h @ W_O_h) · (W_U[IO] − W_U[S])
    where z_h is the pre-projection hidden state for that head at END, sliced
    from c_proj.input (shape [N, seq, n_heads*d_head] in GPT-2 Conv1D convention).

    tokens          : [N, seq]
    end_positions   : [N] — per-example END token index
    io_token_ids    : list[int] length N — token ID of each example's IO name
    s_token_ids     : list[int] length N — token ID of each example's S name
    Returns         : {(layer, head): Tensor[N]} — one scalar per example
    """
    n_layers = len(model.transformer.h)
    n_heads  = model.config.n_head
    d_head   = model.config.n_embd // n_heads
    N        = tokens.size(0)

    W_U    = model.lm_head.weight.detach().cpu().float()   # [vocab, d_model]
    io_ids = torch.tensor(io_token_ids)
    s_ids  = torch.tensor(s_token_ids)
    io_dir = W_U[io_ids] - W_U[s_ids]                     # [N, d_model]

    result: Dict[Tuple[int, int], torch.Tensor] = {}
    for layer in range(n_layers):
        # GPT-2 Conv1D: c_proj.weight shape is [n_heads*d_head, d_model]
        W_O = model.transformer.h[layer].attn.c_proj.weight.detach().cpu().float()
        with model.trace({"input_ids": tokens}):
            z = model.transformer.h[layer].attn.c_proj.input.save()
        # z: [N, seq, n_heads*d_head]
        z_end = z[torch.arange(N), end_positions].cpu().float()   # [N, n_heads*d_head]
        for head in range(n_heads):
            z_h   = z_end[:, head * d_head : (head + 1) * d_head]  # [N, d_head]
            W_O_h = W_O[head * d_head : (head + 1) * d_head, :]    # [d_head, d_model]
            out_h = z_h @ W_O_h                                     # [N, d_model]
            proj  = (out_h * io_dir).sum(-1)                        # [N]
            result[(layer, head)] = proj
        del z
        clear_cache()
    return result


def ov_copy_strength(model, layer: int, head: int) -> float:
    """How strongly head (layer, head)'s OV circuit copies its attended token to the output.

    Computes W_E @ W_V[head] @ W_O[head] @ W_U and returns the mean diagonal score,
    normalised by the mean off-diagonal score. Values > 1 indicate the head copies
    attended tokens; values near 1 indicate no copy preference.

    Pure weight analysis — no forward pass required.
    """
    d_model  = model.config.n_embd
    n_heads  = model.config.n_head
    d_head   = d_model // n_heads

    W_E = model.transformer.wte.weight.detach()              # [vocab, d_model]
    # c_attn projects [d_model] → [3*d_model]; V slice is [2*d_model : 3*d_model]
    W_QKV = model.transformer.h[layer].attn.c_attn.weight.detach()  # [d_model, 3*d_model]
    W_V   = W_QKV[:, 2 * d_model + head * d_head : 2 * d_model + (head + 1) * d_head]  # [d_model, d_head]
    W_O   = model.transformer.h[layer].attn.c_proj.weight.detach()  # [d_model, d_model]
    W_O_h = W_O[head * d_head : (head + 1) * d_head, :]             # [d_head, d_model]
    W_U   = model.lm_head.weight.detach()                           # [vocab, d_model]

    # OV circuit would be [vocab, vocab] — too large to materialise on MPS.
    # Compute diagonal and off-diagonal statistics without the full matrix.
    #
    # Let A = W_E @ W_V @ W_O_h  shape [V, d_model]
    #     B = W_U                 shape [V, d_model]
    # OV = A @ B.T               shape [V, V]
    #
    # Diagonal mean:   mean_i (A[i] · B[i])
    # Total sum:       (sum_i A[i]) · (sum_j B[j])   (outer-product row/col sums)
    # Off-diag mean:   (total_sum - diag_sum) / (V*(V-1))
    V = W_E.size(0)
    A = (W_E @ W_V @ W_O_h).cpu().float()   # [V, d_model]  force CPU+fp32 to avoid MPS limits
    B = W_U.cpu().float()                    # [V, d_model]

    diag_per_tok = (A * B).sum(-1)           # [V]
    diag_sum     = diag_per_tok.sum().item()
    diag_mean    = diag_sum / V

    total_sum    = (A.sum(0) @ B.sum(0)).item()
    off_diag_mean = (total_sum - diag_sum) / (V * (V - 1))

    return diag_mean / (abs(off_diag_mean) + 1e-8)
