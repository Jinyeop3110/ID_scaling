LLAMA_CONFIG = {
        "session_name": "test",
        "session_path": "Data/test", #NFS remote folder?
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
            "layer_idx_list": [0,1,2,5,10,15],  # Convert numpy array to list
            "module_inblock_keys": ['mlp', 'attn', 'block'],
            "module_outblock_keys": [ 'unemb'],
            "save_fp": "torch.float16",
            "save_cache_tensors":True, # -> 
            "save_mean_tensors":True,
            "save_IDs":True,
            "save_IDs_list": ['mle','mind_ml', 'twoNN_f10'],
        },
        'multiprocessing' : True,
        'multiprocessing_num_cpus' : 30,
        'verbose' : True
    }