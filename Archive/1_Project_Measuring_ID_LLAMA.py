#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# %%


#import installed libraries

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
import numpy as np 

from neel_plotly import line, imshow, scatter
import transformer_lens.patching as patching
import circuitsvis as cv
import matplotlib.pyplot as plt
from pathlib import Path


# %%


# Import from local libraries

from load_HookedTransformer import load_HookedTransformer
from utils import *
from datasets import load_dataset
from load_Datasets import load_and_concat_openwebtext


# %%





# %%


torch.set_grad_enabled(False)
#model,tokenizer = load_HookedTransformer('llama-7b')
model,tokenizer = load_HookedTransformer('llama-7b')


# %%


subset=[0]
dataset = load_and_concat_openwebtext(subset)


# %%





# %%





# %%





# %%


def get_last_index(tokens, pad_token_index):
    max_len=tokens.shape[1]
    tokens=tokens[:,1:].cpu()
    
    # Create a boolean mask where the condition is True for the pad_token_index
    pad_mask = (tokens == pad_token_index)

    # However, argmax returns 0 for rows where the condition is never True (i.e., no pad_token_index found).
    # To distinguish between cases where the pad_token_index is at the start (index 0) and not present, 
    # we can use a trick: multiply the mask by a range tensor and use argmax on that.
    # If the pad token is not present in the row, the result will be 0 for that row, since the mask will be all False.

    # Generate a range tensor of the same shape as prompts
    last_index = pad_mask.int().argmax(dim=1)
    last_index[last_index==0]=max_len-1
    
    return last_index.numpy()

'''
tokens=model.to_tokens(dataset.iloc[:1000].text)
text_lengths=get_last_index(tokens)
text_lengths
plt.hist(get_last_index(prompts)+1)

# Add labels and title
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.title('Distribution of Text Lengths')

# Show plot
plt.show()
'''


# %%





# %%





# %%





# %%


project_name = 'final_activation_llama_set0_ctx_1024'
if not os.path.exists(project_name):
    os.mkdir(project_name)
    print(f"Directory '{project_name}' created.")
else:
    print(f"Directory '{project_name}' already exists.")

import os
import torch
from pathlib import Path

num_samples = 100000
batch_size = 6
ctx_len=1024

model.cfg.n_ctx=ctx_len

# Initialize dictionaries to hold final activations for both sets of positions
final_activations = {
    'last0': {}, 'last1': {}, 'last2': {}, 'last3': {}, 'last10': {}, 'last20': {}, 
    'pos0': {}, 'pos5': {}, 'pos10': {}, 'pos50': {}
}

target_keys_substrings = [
    "hook_attn_out",
    "hook_mlp_out",
    "hook_resid_pre",
    "hook_resid_mid",
    "hook_resid_post",
    "ln_final"
]

layer_indices = [0, 1, 2, 8, 15, 23, 29, 30, 31]
target_layers_substrings = [f"blocks.{index}." for index in layer_indices] + ["ln_final"]

# Define specific token positions

names_filter= [key for key in list(model.hook_dict.keys()) if any(substring in key for substring in target_keys_substrings)]
names_filter= [key for key in names_filter if any(substring in key for substring in target_layers_substrings)]

last_token_positions=[0,1,2,3,10,20]
token_positions = [0, 5, 10, 50]  
last_idxs=[]
for i in range(int(np.ceil(num_samples // batch_size))):
    prompts = dataset.iloc[batch_size * i : min(batch_size * (i + 1), num_samples)].text
    tokens=model.to_tokens(prompts.tolist())
    last_idx = get_last_index(tokens, pad_token_index=tokenizer.pad_token_id)
    last_idxs+=last_idx.tolist()
filtered_idx=(np.array(last_idxs)==(ctx_len-1))
dataset=dataset.iloc[np.where(filtered_idx)]

num_samples=len(dataset) 
last_idxs=[]
for i in range(int(np.ceil(num_samples // batch_size))):
    prompts = dataset.iloc[batch_size * i : min(batch_size * (i + 1), num_samples)].text
    
    tokens=model.to_tokens(prompts.tolist())
    #tokens = torch.concatenate([t[:ctx_len] for t in tokens])
    tokens=tokens[:,:ctx_len]
    last_idx = get_last_index(tokens, pad_token_index=tokenizer.pad_token_id)
    _, cache = model.run_with_cache(tokens, names_filter=names_filter)

    # Process each key in the cache
    for key in cache.keys():
        if any(substring in key for substring in target_keys_substrings):

            if hasattr(cache[key], 'shape') and len(cache[key].shape) == 3:
                # Process for last positions relative to last_idx
                for offset, suffix in zip(last_token_positions, ['last0', 'last1', 'last2', 'last3', 'last10', 'last20']):
                    position = -offset
                    slices = [cache[key][j, last_idx[j] + position, :].cpu() for j in range(len(last_idx)) if last_idx[j] + position >= 0]
                    if slices:
                        if key not in final_activations[suffix]:
                            final_activations[suffix][key] = []
                        final_activations[suffix][key].extend(slices)

                # Process for specific token positions
                for position in token_positions:
                    if tokens.shape[1] > position:  # Ensure sequence is long enough
                        slices = [cache[key][j, position, :].cpu() for j in range(tokens.shape[0]) if tokens.shape[1] > position]
                        if slices:
                            suffix = f'pos{position}'
                            if key not in final_activations[suffix]:
                                final_activations[suffix][key] = []
                            final_activations[suffix][key].extend(slices)
    last_idxs=last_idxs+list(last_idx)
    del _
    del cache
    del tokens
    torch.cuda.empty_cache()
    print(f"{batch_size * (i + 1) - 1}th samples done")

# Save the extracted slices
for suffix in final_activations.keys():
    for key, value in final_activations[suffix].items():
        if value: 
            tensor_stack = torch.stack(value, dim=0)
            torch.save(tensor_stack, Path(project_name) / f'{key}.{suffix}.pt')
            
            
torch.save(last_idxs, Path(project_name) / 'last_idxs.pt')

