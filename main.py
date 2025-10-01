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
    find_name_movers,
)

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_gpt2()

# Load single example dataset
ioi_dataset = json.load(open("./data/eng_ioi.json"))
data_batches = get_batches(model, ioi_dataset, n_samples=300, batch_size=300, seed=42)

# Find name mover heads
find_name_movers(model, data_batches, causal_effect_threshold=0.01)
