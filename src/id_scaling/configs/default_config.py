from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FilterAndChunkConfig:
    min_chunks_from_a_document: int = 5
    max_chunks_from_a_document: int = 5

@dataclass
class DatasetConfig:
    dataset_name: str = 'stas/openwebtext-10k' #"monology/pile-uncopyrighted"
    dataset_subset: str = 'train' #'test' # List[int] = field(default_factory=lambda: [0])
    max_dataset_size: int = 100
    filter_and_chunk_config: FilterAndChunkConfig = field(default_factory=FilterAndChunkConfig)

@dataclass
class ModelConfig:
    model_name: str = 'mistralai/Mistral-7B-Instruct-v0.1'
    model_checkpoint: str = "main"
    use_accelerator: bool = False
    module_name_mapping: Dict[str, str] = field(default_factory=lambda: {
        "mlp": "model.layers.{layer}.mlp",
        "attn": "model.layers.{layer}.self_attn",
        "block": "model.layers.{layer}",
        "emb": "model.embed_tokens",
        "unemb": "model.norm",
    })

@dataclass
class CachingConfig:
    layer_idx_list: List[int] = field(default_factory=lambda: [0, 1, 10, 15, 31])
    module_inblock_keys: List[str] = field(default_factory=lambda: ['mlp', 'attn', 'block'])
    module_outblock_keys: List[str] = field(default_factory=lambda: ['unemb'])
    save_fp: str = "torch.float16"
    save_cache_tensors: bool = True
    save_mean_tensors: bool = True
    save_IDs: bool = True
    save_IDs_list: List[str] = field(default_factory=lambda: ['mle', 'mind_ml', 'twoNN_f10'])

@dataclass
class DefaultConfig:
    base_dir: str = None
    session_name: str = "test"
    session_path: str = "/home/gridsan/jsong/physics_dl_shared/ML_JY/ID_scaling/Data/test"
    model_config: ModelConfig = field(default_factory=ModelConfig)
    dataset_config: DatasetConfig = field(default_factory=DatasetConfig)
    ctx_len: int = 1024
    batch_size: int = 2
    cacheing_config: CachingConfig = field(default_factory=CachingConfig)
    multiprocessing: bool = True
    multiprocessing_num_cpus: int = 30
    verbose: bool = True