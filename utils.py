import gc
import torch
from nnsight import LanguageModel


def load_model(name: str = "openai-community/gpt2") -> LanguageModel:
    model = LanguageModel(name, device_map="auto", dispatch=True)
    model.tokenizer.padding_side = "right"
    model.config._attn_implementation = "eager"
    return model


def clear_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()
