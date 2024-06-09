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
model,tokenizer = load_HookedTransformer('gpt2-large')


# %%


subset=[0]
dataset = load_and_concat_openwebtext(subset)


# %%





# %%


def get_last_index(tokens, pad_token_index = 50256):
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





# %%


project_name = 'final_activation_gpt2_large_set0_ctx_1024'
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

if False : 
    last_idxs=[]
    for i in range(int(np.ceil(num_samples // batch_size))):
        prompts = dataset.iloc[batch_size * i : min(batch_size * (i + 1), num_samples)].text
        tokens=model.to_tokens(prompts.tolist())
        last_idx = get_last_index(tokens)
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
        last_idx = get_last_index(tokens)
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
    

    
    

def get_ratios(vectors, n_neighbors):
    try:
        N = len(vectors)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto").fit(vectors)
        distances, indices = nbrs.kneighbors(vectors)
        ratios = np.array(
            [distances[:, i] / distances[:, 1] for i in range(2, n_neighbors)]
        )

    except Exception as e:
        print(f"An error occurred: {e}")
        ratios = None
        N = 0

    return ratios, N


def measure_dimension_kNN(
    vectors, n_neighbors=5, fraction=0.9, plot=False, verbose=False
):
    ratios, N = get_ratios(vectors, n_neighbors)
    try:
        mus = [
            np.sort(ratios[i], axis=None, kind="quicksort") for i in range(n_neighbors - 2)
        ]
        Femp = (np.arange(1, N + 1, dtype=np.float64)) / N

        dims = []
        xs = []
        ys = []
        regrs =[]
        for k, mu in enumerate(mus):
            x = np.log(mu[:-2])
            xs += [x]
            y = -np.log(1 - Femp[:-2] ** (1 / (k + 1)))
            ys += [y]

            npoints = int(np.floor(N * fraction))
            regr = linear_model.LinearRegression(fit_intercept=False)
            regr.fit(x[:npoints, np.newaxis], y[:npoints, np.newaxis])
            if verbose:
                print(
                    "From ratio " + str(k + 2) + " NN estimated dim " + str(regr.coef_[0])
                )
            dims += [regr.coef_[0]]
            regrs += [regr]

        if plot:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.set_title("Log F vs Log mu")
            ax.set_xscale("linear")
            ax.set_yscale("linear")
            for x, y, dim, regr in zip(xs, ys, dims, regrs):
                ax.scatter(x[:npoints], y[:npoints])
                ax.plot(
                    x[:npoints], regr.predict(x[:npoints].reshape((-1, 1))), color="gray"
                )
                ax.text(x[0],y[0], f'Dim {dim}', fontsize=14, color='blue')
        else : 
            fig = None
        #print(x[:npoints],regr.predict(x[:npoints].reshape((-1, 1)) ))
        return xs, ys, dims, regrs, npoints, fig
    except:
        return None, None, None, None, None, None


def measure_dimension_MLE(vectors, n_neighbors=10, plot=False, verbose=False):
    ratios, _ = get_ratios(vectors, n_neighbors)
    logs = np.log(ratios)
    estimates = (n_neighbors - 2) / (logs[:, -1] - np.sum(logs[:, :-1], axis=1))
    dim = np.mean(estimates)
    var = np.var(estimates)
    if verbose:
        print("Dimension MLE: ", dim, " Stddev: ", np.sqrt(var))
    if plot:
        fig, axs = plt.subplots(1, 1)
        axs.hist(estimates, bins=50)

    return dim, var, estimates


if True:
    last_idxs= torch.load(Path(project_name)/'last_idxs.pt')
    
    import itertools
    import pickle
    from sklearn.neighbors import NearestNeighbors
    from sklearn import linear_model
    
    # Use itertools.product to create combinations
    last_token_ind=1023
    fraction=0.1
    keys=[]
    dims_dict={}
    # Print each combination
    for str_i,str_j in list(itertools.product(names_filter, final_activations.keys()))[:]:
        key=f"{str_i}.{str_j}"
        file=key+".pt"
        activation_tensors0 = torch.load(Path(project_name)/file)
        filtered_data_idx=(np.array(last_idxs)==last_token_ind)
        activation_tensors_filtered=activation_tensors0[filtered_data_idx,:]

        xs, ys, dims, regrs, npoints, fig = measure_dimension_kNN(
            activation_tensors_filtered, n_neighbors=5, fraction=fraction, plot=True, verbose=True
        )

        if fig is not None:
            fig.savefig(Path(project_name) / ('plot_ID_'+file.replace(".pt", ".png")))

        dims_dict[key]=dims
        keys+=[key]

    Dimension_analyzed={
        "positions" :list(final_activations.keys()),
        "layer_keys" :target_layers_substrings,
        "layer_ind":layer_indices,
        "names_filter" : names_filter,
        "keys":keys,
        "dims":dims_dict,
        "last_token_ind":last_token_ind,
        "filtered_data_idx":filtered_data_idx,
        "fraction":fraction,
        "sample_num" : len(filtered_data_idx),
        "filtered_sample_num": sum(filtered_data_idx)
    }

    with open(Path(project_name)/'Dimension_analyzed.pkl', 'wb') as f:
        pickle.dump(Dimension_analyzed, f)
