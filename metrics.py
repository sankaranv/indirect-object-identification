import torch
import numpy as np


def logit_diff(model, dataset):

    logit_diffs = []
    for batch in dataset:
        prompts = batch["clean_prompts"]
        correct_answers = batch["correct_answers"]
        incorrect_answers = batch["incorrect_answers"]
        correct_answer_token_ids = torch.tensor(batch["correct_answer_token_ids"])
        incorrect_answer_token_ids = torch.tensor(batch["incorrect_answer_token_ids"])
        batch_logit_diffs = []

        with model.trace() as tracer:
            with tracer.invoke(prompts) as invoker:
                logits = model.lm_head.output
                # Get logits for the next token position (after the prompt)
                next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
                correct_logits = next_token_logits[correct_answer_token_ids]
                incorrect_logits = next_token_logits[incorrect_answer_token_ids]
                logit_diff = correct_logits - incorrect_logits
                batch_logit_diffs.append(logit_diff.mean().item())
        logit_diffs.append(batch_logit_diffs)
    return logit_diffs
