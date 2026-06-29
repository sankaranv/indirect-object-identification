import os
import sys
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ioi")
)

from utils import load_model
from metrics import logit_diff
from ioi_dataset import IOIDataset

torch.set_grad_enabled(False)

model = load_model()
ioi = IOIDataset("mixed", N=300, tokenizer=model.tokenizer, prepend_bos=False)
abc = ioi.gen_flipped_prompts(("IO", "RAND"))
abc = abc.gen_flipped_prompts(("S", "RAND"))
end_pos = ioi.word_idx["end"]
N = len(ioi)

with model.trace({"input_ids": ioi.toks.long()}):
    clean_logits = model.lm_head.output.save()
with model.trace({"input_ids": abc.toks.long()}):
    corr_logits = model.lm_head.output.save()

clean_ld = (
    logit_diff(
        clean_logits.cpu()[torch.arange(N), end_pos],
        ioi.io_tokenIDs,
        ioi.s_tokenIDs,
    )
    .mean()
    .item()
)
corr_ld = (
    logit_diff(
        corr_logits.cpu()[torch.arange(N), end_pos],
        ioi.io_tokenIDs,
        ioi.s_tokenIDs,
    )
    .mean()
    .item()
)

print(f"[Phase 1] Clean logit diff:     {clean_ld:.4f}  (paper: ~3.55)")
print(f"[Phase 1] Corrupted logit diff: {corr_ld:.4f}  (paper: ~-0.38)")
assert clean_ld > 1.0, f"clean LD too low: {clean_ld}"
assert corr_ld < clean_ld
print("[Phase 1] PASS")
