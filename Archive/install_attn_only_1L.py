# import transformer_lens

import sys
from fancy_einsum import einsum

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformers import AutoTokenizer, pipeline, logging, AutoModelForCausalLM, AutoConfig
from transformer_lens import ActivationCache, HookedTransformer

from huggingface_hub import notebook_login
from datasets import Dataset
import pandas as pd
import transformers
import torch
import einops  # Make sure einops is imported

from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching
import circuitsvis as cv

model_paths = {
        #'llama-7b': "../Models/Llama-2-7b-hf",
        #'mistral-7b': "../Models/Mistral-7B-v0.1",
        'attn-only-1l': "../Models/Attn_Only_1L512W_C4_Code",
        'attn-only-2l': "../Models/Attn_Only_2L512W_C4_Code",
}
model_type='attn-only-1l'
tokenizer_path = "../Models/gpt-neox-tokenizer-digits" if "attn-only" in model_type else model_paths[model_type]

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

A=HookedTransformer.from_pretrained(model_type, tokenizer=tokenizer, device="cpu", dtype="float16")
print(A)
