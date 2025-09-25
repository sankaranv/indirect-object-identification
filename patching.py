import torch
from typing import Dict


def logit_diff_metric(
    logits: torch.Tensor, correct_ids: torch.Tensor, incorrect_ids: torch.Tensor
):
    """Default IOI metric: logit(correct) - logit(incorrect) at final position."""
    final = logits[:, -1, :]
    rows = torch.arange(final.size(0), device=final.device)
    return final[rows, correct_ids] - final[rows, incorrect_ids]


def path_patching(model, data_batches, sender_head, receiver_nodes, metric_fn) -> Dict:
    """
    Algorithm 1: Path Patching from the IOI paper.

    Args:
        model: The language model
        data_batches: Dictionary containing treatment and baseline prompts
        sender_head: Sender attention head (layer, head_idx)
        receiver_nodes: List of receiver nodes to patch to
        metric_fn: Function to compute the metric

    Returns:
        Dictionary with path patching results
    """

    results = {}
    n_layers = len(model.transformer.h)
    n_head = model.config.n_head
    d_model = model.config.n_embd
    d_head = d_model // n_head
    sender_layer_idx, sender_head_idx = sender_head

    # Attention heads are all concatenated together in GPT2
    # To get the dimensions of a head output that correspond to one head, we need to slice out its chunk of size d_head
    sender_slice = slice(sender_head_idx * d_head, (sender_head_idx + 1) * d_head)

    for i, batch in enumerate(data_batches):

        # Get data from the batch
        treatment_inputs = batch["treatment_inputs"]
        baseline_inputs = batch["baseline_inputs"]
        correct_answer_token_ids = batch["correct_answer_token_ids"]
        incorrect_answer_token_ids = batch["incorrect_answer_token_ids"]

        # Get activations on the baseline input
        with model.trace(baseline_inputs) as baseline_run_tracer:
            baseline_head_outputs = [
                model.transformer.h[l].attn.c_proj.input.save() for l in range(n_layers)
            ].save()

        # Get activations and compute the metric under treatment
        with model.trace(treatment_inputs) as treatment_run_tracer:
            treatment_head_outputs = [
                model.transformer.h[l].attn.c_proj.input.save() for l in range(n_layers)
            ].save()
            treatment_logits = model.lm_head.output.save()

        treatment_metric = metric_fn(
            treatment_logits.cpu(),
            correct_answer_token_ids,
            incorrect_answer_token_ids,
        )

        # Get activations of the receiver nodes after patching the sender head using activations from the baseline run
        receiver_activations_to_patch = {}
        with model.trace(treatment_inputs) as sender_patch_tracer:

            # In NNsight we always apply interventions in forward pass order
            for l in range(n_layers):

                # Patch the sender head activations with those from the baseline run
                if l == sender_layer_idx:
                    model.transformer.h[l].attn.c_proj.input[..., sender_slice] = (
                        baseline_head_outputs[l][..., sender_slice]
                    )
                # Freeze all remaining heads to their activations under treatment
                else:
                    model.transformer.h[l].attn.c_proj.input[...] = (
                        treatment_head_outputs[l]
                    )

            # Save the activations of the receiver node to use in the next run
            # Receivers can either be "logits", a tuple (layer_idx, "mlp"), or a tuple (layer_idx, head_idx)
            for receiver in receiver_nodes:
                if receiver == "logits":
                    receiver_activations_to_patch[receiver] = (
                        model.transformer.ln_f.input.save()
                    )
                elif isinstance(receiver, tuple) and receiver[1] == "mlp":
                    layer_idx = receiver[0]
                    receiver_activations_to_patch[receiver] = model.transformer.h[
                        layer_idx
                    ].mlp.c_fc.input.save()
                elif isinstance(receiver, tuple) and isinstance(receiver[1], int):
                    layer_idx, head_idx = receiver
                    receiver_activations_to_patch[receiver] = model.transformer.h[
                        layer_idx
                    ].attn.c_proj.input.save()
                else:
                    raise NotImplementedError

        # Patch receiver nodes with the activations we saved from the previous run, and complete the forward pass
        with model.trace(treatment_inputs) as receiver_patch_tracer:
            for receiver, activation in receiver_activations_to_patch.items():
                if receiver == "logits":
                    model.transformer.ln_f.input[...] = activation

                elif isinstance(receiver, tuple) and receiver[1] == "mlp":
                    layer_idx = receiver[0]
                    model.transformer.h[layer_idx].mlp.c_fc.input[...] = activation

                elif isinstance(receiver, tuple) and isinstance(receiver[1], int):
                    layer_idx, head_idx = receiver
                    target = model.transformer.h[layer_idx].attn.c_proj.input
                    head_slice = slice(head_idx * d_head, (head_idx + 1) * d_head)
                    target[..., head_slice] = activation[..., head_slice]
                else:
                    raise NotImplementedError

            # Save the logits from the patched run
            patched_logits = model.lm_head.output.save()

        # Obtain the value of the metric under the patched run
        patched_metric = metric_fn(
            patched_logits.cpu(),
            correct_answer_token_ids,
            incorrect_answer_token_ids,
        )

        metric_diff = treatment_metric - patched_metric

        # Report the metric values per prompt along with mean and standard deviation for the batch
        results[i] = {
            "treatment_metric_values": treatment_metric.cpu().tolist(),
            "treatment_metric_mean": treatment_metric.mean().item(),
            "treatment_metric_std": (
                treatment_metric.std().item() if treatment_metric.numel() > 1 else None
            ),
            "patched_metric_values": patched_metric.cpu().tolist(),
            "patched_metric_mean": patched_metric.mean().item(),
            "patched_metric_std": (
                patched_metric.std().item() if patched_metric.numel() > 1 else None
            ),
            "metric_diff_values": metric_diff.cpu().tolist(),
            "metric_diff_mean": metric_diff.mean().item(),
            "metric_diff_std": (
                metric_diff.std().item() if metric_diff.numel() > 1 else None
            ),
        }

    return results
