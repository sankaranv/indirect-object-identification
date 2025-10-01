from utils import *
import json
from metrics import (
    logit_diff,
)  # TODO - there are two different logit_diff functions right now
from patching import path_patching, logit_diff_metric
import torch
from experiments.name_movers import (
    compute_head_to_logit_effects,
    compute_attention_probs_on_input_tokens,
    compute_logit_lens_on_target_tokens,
    copy_score,
)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_gpt2()

# Load single example dataset
ioi_dataset = json.load(open("./data/eng_ioi.json"))
data_batches = get_batches(model, ioi_dataset, n_samples=300, batch_size=300, seed=42)
# logit_diff = logit_diff(model, data_batches)
# print(logit_diff)
# print(data_batches[0])

# sender_head = (5, 3)
# receiver_nodes = ["logits"]
# results = path_patching(
#     model, data_batches, sender_head, receiver_nodes, logit_diff_metric
# )
# print(json.dumps(results, indent=4))

# head_effects = compute_head_to_logit_effects(model, data_batches, logit_diff_metric)
# print(head_effects)

# io_head_attn_probs = compute_attention_probs_on_input_tokens(
#     model, data_batches, key_token="IO", query_token="END"
# )
# s2_head_attn_probs = compute_attention_probs_on_input_tokens(
#     model, data_batches, key_token="S2", query_token="END"
# )

# name_head_attn_probs = {
#     key: io_head_attn_probs[key] + s2_head_attn_probs[key]
#     for key in io_head_attn_probs.keys()
# }
# print(name_head_attn_probs)

# logit_projections = compute_logit_lens_on_target_tokens(
#     model, data_batches, target_token="IO", write_position=-1
# )
# print(logit_projections)

copy_scores = copy_score(model, data_batches, target_token_type=["IO", "S2"])
print(sorted(copy_scores.items(), key=lambda x: -x[1])[:10])
