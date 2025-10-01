from patching import path_patching
from typing import List, Dict, Callable, Tuple, Union
from nnsight import LanguageModel
import torch
from tqdm import tqdm
from utils import clear_cache
import math
import einops


def compute_head_to_logit_effects(
    model: LanguageModel,
    data_batches: List[Dict],
    metric_fn: Callable,
) -> Dict[Tuple[int, int], float]:
    """
    Compute the direct effect of every attention head on the logits using path patching.

    Args:
        model: NNsight LanguageModel (this code is written for GPT2)
        data_batches: List of dicts containing baseline and treatment inputs
        metric_fn: Function to compute logit difference or other scalar metric

    Returns:
        Dictionary mapping (layer_idx, head_idx) -> average metric drop
        when path patching that head to logits.
    """

    n_layers = len(model.transformer.h)
    n_head = model.config.n_head
    head_effects = {}

    for layer in range(n_layers):
        for head in tqdm(range(n_head)):

            # We use path patching to compute the effect from (layer, head) to the logits
            sender_head = (layer, head)
            receiver_nodes = ["logits"]
            results = path_patching(
                model,
                data_batches,
                sender_head=sender_head,
                receiver_nodes=receiver_nodes,
                metric_fn=metric_fn,
            )

            # Aggregate causal effects across batches
            all_diffs = []
            for res in results.values():
                all_diffs.extend(res["metric_diff_values"])
            head_effects[(layer, head)] = (
                torch.tensor(all_diffs, dtype=torch.float32).mean().item()
            )

            # Get rid of the results tensor to free up memory
            del results
            clear_cache()

    return head_effects


def compute_attention_probs_on_input_tokens(
    model: LanguageModel,
    data_batches: List[Dict],
    key_token: str,
    query_token: str,
) -> Dict[Tuple[int, int], float]:
    """
    Compute average attention probability of each head at the query position on the specified target tokens.
    """
    n_layers = len(model.transformer.h)
    n_head = model.config.n_head

    # Accumulate sums and counts to compute running mean
    head_sums = {(l, h): 0.0 for l in range(n_layers) for h in range(n_head)}
    head_counts = {(l, h): 0 for l in range(n_layers) for h in range(n_head)}

    # Compute the average attention probability of each head at the query position on the specified target tokens
    for batch in data_batches:
        treatment_inputs = batch["treatment_inputs"]
        key_positions = batch["important_token_positions"][key_token]
        query_positions = batch["important_token_positions"][query_token]

        # Run a forward pass on treatment inputs to collect attention probabilities
        with model.trace(treatment_inputs) as tracer:
            attn_weights = {}.save()
            for layer in range(n_layers):
                attn_weights[layer] = (
                    tracer.model.transformer.h[layer]
                    .attn.source.attention_interface_0.output[1]
                    .save()
                )

        # Aggregate the attention probabilities across batches
        for layer, layer_attn in attn_weights.items():
            batch_size = layer_attn.shape[0]
            for i in range(batch_size):
                query_pos = query_positions[i]
                key_pos = key_positions[i]
                for head in range(n_head):
                    head_sums[(layer, head)] += layer_attn[
                        i, head, query_pos, key_pos
                    ].item()
                    head_counts[(layer, head)] += 1

        # Clean up tensors from the NNsight trace
        del attn_weights
        clear_cache()

    head_attn_probs = {
        (layer, head): head_sums[(layer, head)] / head_counts[(layer, head)]
        for layer in range(n_layers)
        for head in range(n_head)
    }

    return head_attn_probs


def compute_logit_lens_on_target_tokens(
    model: LanguageModel,
    data_batches: List[Dict],
    target_token: str,
    write_position: int,
) -> Dict[Tuple[int, int], float]:
    """
    Compute the average logit lens score for the target token at the write position.
    """

    # Get the unembedding vector for the target token
    tokenizer = model.tokenizer
    target_token_id = tokenizer.encode(target_token)[0]
    W_U = model.lm_head.weight
    w_target = W_U[target_token_id].to(model.device)

    n_layers = len(model.transformer.h)
    n_head = model.config.n_head
    d_head = model.config.hidden_size // n_head

    logit_projection_sums = {
        l: torch.zeros(n_head).to(model.device) for l in range(n_layers)
    }
    counts = {l: torch.zeros(n_head).to(model.device) for l in range(n_layers)}

    for batch in data_batches:

        treatment_inputs = batch["treatment_inputs"]
        batch_size = len(treatment_inputs["input_ids"])

        # Get the final output of the attention module in each layer before it is added to the residual stream
        with model.trace(treatment_inputs) as tracer:
            layer_outputs = {
                l: model.transformer.h[l].attn.output[0] for l in range(n_layers)
            }.save()

        # For each layer, compute the logit projection of the attention head output on the target token
        for layer, layer_output in layer_outputs.items():

            # Obtain the output of the attention module at the write position
            heads_output = layer_output[:, write_position, :].reshape(
                batch_size, n_head, d_head
            )

            # Project the layer output through the unembedding matrix and obtain the component corresponding to the target token
            w_heads = w_target.reshape(n_head, d_head)
            logit_projection_vals = einops.einsum(
                heads_output,
                w_heads,
                "batch n_head d_head, n_head d_head -> batch n_head",
            )

            # Aggregate the logit projections across batches
            logit_projection_sums[layer] += logit_projection_vals.sum(dim=0)
            counts[layer] += logit_projection_vals.shape[0]

        del layer_outputs
        clear_cache()

    logit_projections = {}
    for layer in range(n_layers):
        logit_projection_sums[layer] /= counts[layer]
        for head in range(n_head):
            logit_projections[(layer, head)] = logit_projection_sums[layer][head].item()

    del logit_projection_sums, counts
    clear_cache()

    return logit_projections


def copy_score(
    model: LanguageModel,
    data_batches: List[Dict],
    target_token_type: Union[str, List[str]] = ["IO", "S2"],
    top_k: int = 5,
) -> Dict[Tuple[int, int], float]:
    """
    The copy score is obtained using the state of the residual stream at the target token position after the first MLP layer
    We project the residual stream through the OV matrix of a head and apply the logit lens projection to obtain the top-k logits.
    The fraction of times the top-k logits contain the target token is the copy score.
    """

    if isinstance(target_token_type, str):
        if target_token_type not in ["IO", "S", "S1", "S2"]:
            raise ValueError(
                f"Copy score is not implemented for token type {target_token_type}"
            )
        target_token_type = [target_token_type]
    else:
        for tok in target_token_type:
            if tok not in ["IO", "S", "S1", "S2"]:
                raise ValueError(f"Copy score is not implemented for token type {tok}")

    W_U = model.lm_head.weight

    n_layers = len(model.transformer.h)
    n_head = model.config.n_head
    d_model = model.config.hidden_size
    d_head = d_model // n_head

    successes = {(l, h): 0 for l in range(n_layers) for h in range(n_head)}
    num_samples = 0

    for batch in data_batches:

        treatment_inputs = batch["treatment_inputs"]
        batch_size = len(treatment_inputs["input_ids"])
        num_samples += batch_size

        # Get the positions of the target tokens in the batch
        positions = {
            tok: torch.as_tensor(
                batch["important_token_positions"][tok], device=model.device
            )
            for tok in target_token_type
        }

        # Get the token ids for each prompt in the batch corresponding to the target token types
        token_ids = {
            tok: (
                torch.as_tensor(batch["correct_answer_token_ids"], device=model.device)
                if tok == "IO"
                else torch.as_tensor(
                    batch["incorrect_answer_token_ids"], device=model.device
                )
            )
            for tok in target_token_type
        }

        # Get the residual stream after the first layer
        with model.trace(treatment_inputs) as tracer:
            residual_stream = tracer.model.transformer.h[0].output[0].save()

        for l in range(n_layers):

            # Apply layer normalization to the residual stream
            normalized_residual_stream = model.transformer.h[l].ln_1(residual_stream)

            # Get value and output matrices for the layer
            W_O = model.transformer.h[l].attn.c_proj.weight
            W_V = model.transformer.h[l].attn.c_attn.weight[
                :, 2 * d_model : 3 * d_model
            ]
            b_V = model.transformer.h[l].attn.c_attn.bias[2 * d_model : 3 * d_model]

            for h in range(n_head):

                # Slice out the value and output vectors for the head
                W_V_head = W_V[:, h * d_head : (h + 1) * d_head]
                b_V_head = b_V[h * d_head : (h + 1) * d_head]
                W_O_head = W_O[h * d_head : (h + 1) * d_head, :]

                # Initialize a mask to track whether the top-k logits contain any of the target tokens
                mask_success = torch.zeros(
                    batch_size, dtype=torch.bool, device=model.device
                )

                for target_token in target_token_type:

                    # Slice out the residual stream at the target token position
                    residual_at_target_token = normalized_residual_stream[
                        torch.arange(batch_size), positions[target_token], :
                    ]

                    # Project the residual stream at the target token position through the OV matrix of the head
                    # This simulates what would happen if the head attended perfectly to that token
                    v_out = (
                        (residual_at_target_token @ W_V_head) + b_V_head
                    ) @ W_O_head

                    # Project the output of the OV matrix through the unembedding matrix and obtain the top-k logits
                    logits = model.transformer.ln_f(v_out) @ W_U.T
                    topk = logits.topk(top_k, dim=-1).indices

                    # Count the number of times the top-k logits contain the target token
                    mask_success |= (topk == token_ids[target_token].unsqueeze(1)).any(
                        dim=-1
                    )
                successes[(l, h)] += mask_success.sum().int().item()

        # Clean up tensors from the NNsight trace
        del residual_stream
        clear_cache()

    # The copy score is the fraction of times the top-k logits contain the target token
    copy_scores = {
        (l, h): successes[(l, h)] / num_samples
        for l in range(n_layers)
        for h in range(n_head)
    }

    return copy_scores


def find_name_movers():
    pass
