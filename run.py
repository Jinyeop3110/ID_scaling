import os
import time
import argparse
import logging

from accelerate import Accelerator
import pandas as pd
import torch
from rich.logging import RichHandler

from id_scaling.model import load_model_and_tokenizer
from id_scaling.data import load_dataset, process_dataset
from id_scaling.utils import *
from id_scaling.cache import *
from id_scaling.configs.default_config import DefaultConfig
from id_scaling.utils.config_utils import update_config_from_yaml
from id_scaling.utils.model_utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


######### LOGGING #########
logger = logging.getLogger("rich")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


# Function to process each batch, run the model, and calculate entropy
def run_batch_with_cache(model, batch, config):
    input_token_ids = torch.tensor(batch['tokens'])

    len_list = get_len_list(input_token_ids, -1, config.ctx_len)
    df_metadata = pd.DataFrame({key: value for key, value in batch.items() if key != 'text'})
    df_metadata['len'] = len_list
    
    if torch.cuda.is_available():
        input_token_ids = input_token_ids.cuda()

    with torch.no_grad():
        outputs = model(input_token_ids, use_cache=False)
        logits = outputs.logits
        del outputs

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_token_ids[:, 1:].contiguous()

        loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss.view(shift_logits.size(0), shift_logits.size(1))

        # Calculate entropy from logits
        probs = torch.softmax(shift_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs+1e-9), dim=-1)
        # entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)
        del probs
        torch.cuda.empty_cache()

    return df_metadata, loss.detach(), entropy.detach()


def main(config: DefaultConfig):
    ######### VARIABLES #########
    session_path = config.session_path  # Update this to your desired path
    batch_size = config.batch_size
    
    ######### MODEL #########
    logger.info('Loading model and tokenizer...')
    model, tokenizer = load_model_and_tokenizer(config.model_config)
    if torch.cuda.is_available():
        if config.model_config.use_accelerator:
                accelerator = Accelerator(mixed_precision='fp16') 
                model = accelerator.prepare(model)
        else:
            if torch.cuda.is_available():
                model.cuda()
    model.eval()


    ######### DATASET #########
    logger.info('Loading dataset...')
    dataset = load_dataset(config.dataset_config, supercloud=config.supercloud)
    logging.info(f'Original dataset size: {len(dataset)}')

    ######### DATASET PREPROCESSING #########
    processed_dataset = process_dataset(dataset, tokenizer, config.ctx_len, config.dataset_config.filter_and_chunk_config, return_text=True, offset=50)
    logging.info(f'Processed dataset size: {len(processed_dataset)}')

    ######### SETTING MODULE NAMES #########
    module_name_mapping, module_name_keys = create_module_names(config.model_config.module_name_mapping, config.cacheing_config.layer_idx_list, config.cacheing_config.module_inblock_keys, config.cacheing_config.module_outblock_keys)
    logging.info(f'module name mapping: {module_name_mapping}')
    logging.info(f'module name keys: {module_name_keys}')

    ######### RUN EXPERIMENT #########
    logger.info('Starting experiment...')
    cache_manager = CacheManager(session_path, \
                                 save_cache_tensors=config.cacheing_config.save_cache_tensors, \
                                 save_mean_tensors=config.cacheing_config.save_mean_tensors, \
                                 save_ids = config.cacheing_config.save_IDs, \
                                 num_samples=len(processed_dataset), \
                                 ctx_len=config.ctx_len, \
                                 embedding_dims=model.config.hidden_size, \
                                 multiprocessing=config.multiprocessing, \
                                 num_cpus=config.multiprocessing_num_cpus, \
                                 verbose=config.verbose)

    # Main function to run the experiment
    os.makedirs(session_path, exist_ok=True)
    save_config(config, session_path)

    # Register hooks
    hooks = register_hooks(model, module_name_mapping)

    # Iterate over the dataset with batch size
    for i in range(0, len(processed_dataset), batch_size):

        start_time = time.time()

        batch = processed_dataset[i:i + batch_size]
        df_metadata, loss, entropy = run_batch_with_cache(model, batch, config)

        forward_pass_time = time.time()
        logging.info(f"Forward pass time consumed: {forward_pass_time - start_time:.4f} seconds")

        # Save the tensors to the cache (if enabled)
        cache_manager.save_cache_tensors(CACHE)
        cache_manager.save_mean_tensors(CACHE)
        cache_manager.save_IDs(CACHE, config.cacheing_config.save_IDs_list)

        # Save loss, entropy, and metadata without individual time checks
        cache_manager.save_loss(loss)
        cache_manager.save_entropy(entropy)
        cache_manager.save_metadata(df_metadata)

        total_end_time = time.time()
        print(f"Total cache save time consumed: {total_end_time - forward_pass_time:.4f} seconds")
        print(f"Total process time consumed: {total_end_time - start_time:.4f} seconds")

        # Increment the cache_manager and sanity check
        cache_manager.check_and_increment_index(increment=len(df_metadata))
        clear_gpu_memory(CACHE)
        print(f"{i} to {i+batch_size-1} done / vectors, loss and entropy are calculated and saved.")
    
    # Remove hooks after processing
    remove_hooks(hooks)

if __name__ == "__main__":

    # Create the parser
    parser = argparse.ArgumentParser(description='Load a configuration file.')

    # Add the arguments
    parser.add_argument('--config_path', type=str, help='The path to the configuration file')
    parser.add_argument('--supercloud', action='store_true', help='Use the supercloud setup', default=False)

    # Parse the arguments
    args = parser.parse_args()

    config = DefaultConfig()

    # If a configuration file was provided, load it and overwrite the default configuration
    if args.config_path is not None:
        logger.info(f'Updating configuration from file {args.config_path}...')
        update_config_from_yaml(config, args.config_path)

    # If the supercloud flag is set, update the configuration
    if args.supercloud:
        logger.info('Updating configuration for supercloud...')
        config.supercloud = True
    else:
        config.supercloud = False

    main(config)
