# ID_scaling

## Description

This project provides the basis for ID scaling code. It is designed to save and analyze experiments that explore the in-context representation manifold structure and intrinsic dimensionality.

## Installation

```{bash}
mamba env create -n id -f environment.yml
mamba activate id
pip install -e .
```

One can use `conda` instead of `mamba`.

This project primarily consists of two parts: `Run_experiment.py` and `JobSubmission.ipynb`.

### Run_experiment

`Run_experiment.py` is the main script for running the experiments. It takes care of initializing the experiment, running the experiment, and saving the results. To run an experiment, you can use the following command:

## Configuration
This project uses a configuration dictionary to control various aspects of the experiments. Here's an explanation of the default configuration:

### Example of Configuration (Llama2-7b)

```python
default_config = {
    "session_name": "test",
    "session_path": "/home/gridsan/jsong/physics_dl_shared/ML_JY/ID_scaling/Data/test",
    "model_config": {
        "model_name" :'llama-7b',
        "model_checkpoint": "main",
        "use_accelerator" : False,
        "module_name_mapping":{
            "mlp":"model.layers.{layer}.mlp",
            "attn":"model.layers.{layer}.self_attn",
            "block":"model.layers.{layer}",
            "emb":"model.embed_tokens",
            "unemb":"model.norm",
        }
    },
    "dataset_config": {
        "dataset_name" : "pile_uncopyrighted_parquet_test",
        "dataset_subset" : [0],
        "max_dataset_size" : 1000,
        "filter_and_chunk_config" : {
            "min_chunks_from_a_document" : 5,
            "max_chunks_from_a_document" : 5
        }
    },
    "ctx_len": 1024,
    "batch_size": 2,
    "cacheing_config" : {
        "layer_idx_list": [0,1,2,5],
        "module_inblock_keys": ['mlp', 'attn', 'block'],
        "module_outblock_keys": ['unemb'],
        "save_fp": "torch.float16",
        "save_cache_tensors":True,
        "save_mean_tensors":True,
        "save_IDs":True,
        "save_IDs_list": ['mle','mind_ml', 'twoNN_f10'],
    },
    'multiprocessing' : True,
    'multiprocessing_num_cpus' : 20,
    'verbose' : True
}

### General Settings

- `session_name`: Name of the session, used for the saving directory.
- `session_path`: Path for the saving directory.
- `ctx_len`: Context length for the experiment.
- `batch_size`: Batch size for the experiment.
- `multiprocessing`: Boolean flag to enable or disable multiprocessing.
- `multiprocessing_num_cpus`: Number of CPUs to use if multiprocessing is enabled.
- `verbose`: Boolean flag to control detailed output during the experiment.

### Model Configuration

- `model_name`: Name of the model to be used (e.g., 'llama-7b').
- `model_checkpoint`: Model checkpoint to use.
- `use_accelerator`: Boolean flag to enable or disable the use of an accelerator.
- `module_name_mapping`: Dictionary mapping module names to their paths in the model architecture.

### Dataset Configuration

- `dataset_name`: Name of the dataset to be used.
- `dataset_subset`: Subset of the dataset to use.
- `max_dataset_size`: Maximum size of the dataset.
- `filter_and_chunk_config`: Configuration for filtering and chunking the dataset.

    - `min_chunks_from_a_document`: Minimum number of chunks to extract from a document.
    - `max_chunks_from_a_document`: Maximum number of chunks to extract from a document.

### Caching Configuration

- `layer_idx_list`: List of layer indices to cache.
- `module_inblock_keys`: Keys for in-block modules to cache.
- `module_outblock_keys`: Keys for out-block modules to cache.
- `save_fp`: Floating point precision for saving (e.g., 'torch.float16').
- `save_cache_tensors`: Boolean flag to save cache tensors (whole IC manifold).
- `save_mean_tensors`: Boolean flag to save mean cache tensors (center of IC manifold).
- `save_IDs`: Boolean flag to save IDs (IDs of IC manifolds).
- `save_IDs_list`: List of ID types to save.
