from .default_config import *

LLAMA_CONFIG = DefaultConfig(
    model_config=ModelConfig(
        model_name='llama-7b',
        model_checkpoint='main',
        use_accelerator=False,
        module_name_mapping={
            "mlp": "model.layers.{layer}.mlp",
            "attn": "model.layers.{layer}.self_attn",
            "block": "model.layers.{layer}",
            "emb": "model.embed_tokens",
            "unemb": "model.norm",
        }
    ),
    ctx_len=1024,
    batch_size=2,
    cacheing_config=CachingConfig(
        layer_idx_list=[0, 1, 2, 5, 10, 15],
        module_inblock_keys=['mlp', 'attn', 'block'],
        module_outblock_keys=['unemb'],
        save_fp='torch.float16',
        save_cache_tensors=True,
        save_mean_tensors=True,
        save_IDs=True,
        save_IDs_list=['mle', 'mind_ml', 'twoNN_f10']
    ),
)