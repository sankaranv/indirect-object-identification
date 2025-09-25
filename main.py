from utils import *
import json
from metrics import (
    logit_diff,
)  # TODO - there are two different logit_diff functions right now
from patching import path_patching, logit_diff_metric
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_gpt2()

# Load single example dataset
ioi_dataset = json.load(open("./data/eng_ioi.json"))
data_batches = get_batches(model, ioi_dataset, n_samples=300, batch_size=50, seed=42)
# logit_diff = logit_diff(model, data_batches)
# print(logit_diff)
# print(data_batches[0])

sender_head = (5, 3)
receiver_nodes = ["logits"]
results = path_patching(
    model, data_batches, sender_head, receiver_nodes, logit_diff_metric
)
print(json.dumps(results, indent=4))
