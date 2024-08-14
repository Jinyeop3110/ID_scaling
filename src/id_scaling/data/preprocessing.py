import logging

import torch
from datasets import Dataset
import numpy as np
import pandas as pd
from rich.logging import RichHandler


######### LOGGING #########
logger = logging.getLogger("rich")
logging.basicConfig(level=logging.INFO, handlers=[RichHandler()])


# dataset preprocessing (tokenization, filtering by min length, chunking)
def process_dataset(dataset, tokenizer, ctx_len, filtering_config, return_text = False, offset=0):

    # tokenization fn
    def tokenize(item, tokenizer, return_text=False):
        tokens = tokenizer(item['text'], truncation=False, return_tensors='pt')
        item['tokens'] = tokens['input_ids'][0]

        return item

    # checking min token length
    def check_min_token_length(item, tokenizer, tokens_min_length):
        tokens = tokenizer(item['text'], truncation=False, return_tensors='pt')

        return tokens['input_ids'].shape[1] >= tokens_min_length

    # chunking by chunk_size (batch processing)
    def chunk_text(batch, tokenizer, chunk_size, max_chunk_idx, offset, return_text=False):
        has_bos = hasattr(tokenizer, 'bos_token_id') and tokenizer.bos_token_id is not None and tokenizer.add_bos_token
        start = 1 if has_bos else 0

        chunks_with_meta = {k: [] for k in batch.keys()}

        for b in range(len(batch['tokens'])):
            tokens = batch['tokens'][b]
            chunks = []

            # Manually index the tensor with the desired offset
            for i in range(start, len(tokens), chunk_size+offset):
                chunk = tokens[i:i + chunk_size]
                chunks.append(chunk)

                if len(chunks) > max_chunk_idx:
                    break
            
            chunks_with_meta['tokens'].extend(chunks)
            if return_text:
                chunks_with_meta['text'].extend([tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks])

            if 'id' in batch:
                chunks_with_meta['id'].extend([f"{batch[b]['id']}-{j}" for j in range(len(chunks))])
                
        return chunks_with_meta


    # tokenize the dataset
    dataset = dataset.map(lambda x: tokenize(x, tokenizer), num_proc=64)
    
    # filter the dataset based on minimum token length
    dataset = dataset.filter(lambda x: check_min_token_length(x, tokenizer, ctx_len*(filtering_config.min_chunks_from_a_document+1)), num_proc=64)
    logging.info('Filtered dataset size: {}'.format(len(dataset)))

    # chunk the dataset
    dataset = dataset.map(lambda x: chunk_text(x, tokenizer, ctx_len, filtering_config.max_chunks_from_a_document, True, offset), batched=True, num_proc=1)
    logging.info('Chunked dataset size: {}'.format(len(dataset)))

    size = min(len(dataset), 10000)
    return dataset.select(np.arange(size))