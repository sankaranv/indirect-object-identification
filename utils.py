from nnsight import LanguageModel
import random
import numpy as np
import json


def load_gpt2():
    model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
    return model


def get_important_token_positions(model, dataset):
    important_token_positions = {
        "S1": [],
        "IO": [],
        "S2": [],
        "PLACE": [],
        "OBJECT": [],
        "END": -1,
    }

    for i, datapoint in enumerate(dataset):

        prompt = datapoint["clean_prefix"]
        correct_answer = datapoint["clean_answer"]
        incorrect_answer = datapoint["patch_answer"]
        encoded_tokens = model.tokenizer.encode(prompt, add_special_tokens=False)

        # Decode each token
        tokens = [model.tokenizer.decode([t]) for t in encoded_tokens]

        s_tokens = [f"{incorrect_answer}", f" {incorrect_answer}"]
        s_token_positions = [i for i, t in enumerate(tokens) if t in s_tokens]
        assert len(s_token_positions) == 2

        io_tokens = [f"{correct_answer}", f" {correct_answer}"]
        io_token_positions = [i for i, t in enumerate(tokens) if t in io_tokens]
        assert len(io_token_positions) == 1

        important_token_positions["S1"].append(s_token_positions[0])
        important_token_positions["S2"].append(s_token_positions[1])
        important_token_positions["IO"].append(io_token_positions[0])

    return important_token_positions


def get_answer_token_ids(model, dataset):
    """
    Returns the token ids of the correct and incorrect answers for each datapoint in the dataset.
    """
    answer_token_ids = {"correct": [], "incorrect": []}
    for datapoint in dataset:
        correct_answer = datapoint["clean_answer"]
        incorrect_answer = datapoint["patch_answer"]

        # Tokenize the correct and incorrect answers using the model's tokenizer
        correct_token_id = model.tokenizer.encode(
            correct_answer, add_special_tokens=False
        )[0]
        incorrect_token_id = model.tokenizer.encode(
            incorrect_answer, add_special_tokens=False
        )[0]

        answer_token_ids["correct"].append(correct_token_id)
        answer_token_ids["incorrect"].append(incorrect_token_id)

    return answer_token_ids


def corrupt_ioi_prompt(model, prompt, correct_answer, s2_position):

    encoded_tokens = model.tokenizer.encode(prompt, add_special_tokens=False)
    tokens = [model.tokenizer.decode([t]) for t in encoded_tokens]
    tokens[s2_position] = correct_answer
    baseline_prompt = "".join(tokens)
    return baseline_prompt


def get_batches(
    model, dataset, n_samples, batch_size, seed=42, flip_answer_for_baseline=False
):
    prompt_idxs = list(range(len(dataset)))
    random.seed(seed)
    random.shuffle(prompt_idxs)

    treatment_prompts = [item["clean_prefix"] for item in dataset]
    correct_answers = [item["clean_answer"] for item in dataset]
    incorrect_answers = [item["patch_answer"] for item in dataset]
    answer_token_ids = get_answer_token_ids(model, dataset)
    important_token_positions = get_important_token_positions(model, dataset)

    if flip_answer_for_baseline:
        # Generate baseline prompts by flipping the answer for the treatment prompt
        baseline_prompts = [
            corrupt_ioi_prompt(
                model,
                treatment_prompts[i],
                correct_answers[i],
                important_token_positions["S2"][i],
            )
            for i in range(len(treatment_prompts))
        ]
    else:
        # Use baseline prompts provided in the dataset
        baseline_prompts = [item["patch_prefix"] for item in dataset]

    # Find the set of prompt lengths and counts for each prompt length
    prompt_lengths = list(
        set(
            [
                len(model.tokenizer.encode(p, add_special_tokens=False))
                for p in treatment_prompts
            ]
        )
    )
    counts = [
        len(
            [
                p
                for p in treatment_prompts
                if len(model.tokenizer.encode(p, add_special_tokens=False)) == l
            ]
        )
        for l in prompt_lengths
    ]

    # Pick the most frequently occurring length
    assert (
        max(counts) >= n_samples
    ), f"Cannot produce dataset of size {n_samples} with fixed prompt length"
    fixed_len = prompt_lengths[np.argmax(counts)]
    n_batches = n_samples // batch_size
    data_batches = []
    sampled_prompt_idxs = [
        i
        for i in prompt_idxs
        if len(model.tokenizer.encode(treatment_prompts[i], add_special_tokens=False))
        == fixed_len
    ]

    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_indices = sampled_prompt_idxs[start_idx:end_idx]

        treatment_prompts_text = [treatment_prompts[idx] for idx in batch_indices]
        baseline_prompts_text = [baseline_prompts[idx] for idx in batch_indices]

        # Tokenize the prompts and save them as dicts
        treatment_inputs = model.tokenizer(
            treatment_prompts_text, return_tensors="pt", padding=True
        )
        baseline_inputs = model.tokenizer(
            baseline_prompts_text, return_tensors="pt", padding=True
        )
        # device = next(model.parameters()).device
        # treatment_inputs = {k: v.to(device) for k, v in treatment_inputs.items()}
        # baseline_inputs = {k: v.to(device) for k, v in baseline_inputs.items()}

        # Prepare the batch
        batch = {
            "treatment_prompts_text": treatment_prompts_text,
            "baseline_prompts_text": baseline_prompts_text,
            "treatment_inputs": treatment_inputs,
            "baseline_inputs": baseline_inputs,
            "correct_answers": [correct_answers[idx] for idx in batch_indices],
            "incorrect_answers": [incorrect_answers[idx] for idx in batch_indices],
            "correct_answer_token_ids": [
                answer_token_ids["correct"][idx] for idx in batch_indices
            ],
            "incorrect_answer_token_ids": [
                answer_token_ids["incorrect"][idx] for idx in batch_indices
            ],
        }
        data_batches.append(batch)

    return data_batches


def load_dataset(data_path="./data/eng_ioi.json"):
    with open(data_path, "r") as f:
        dataset = json.load(f)
    return dataset
