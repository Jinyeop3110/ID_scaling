import pandas as pd
import os
from types import SimpleNamespace
from datasets import Dataset

import gzip
import json
import os
from datasets import Dataset, Features, Value
import zstandard as zstd
import io


def load_Datasets(dataset_config_dict):
    dataset_config = SimpleNamespace(**dataset_config_dict)

    if dataset_config.name == 'openwebtext_parquet':
        dataset = load_and_concat_openwebtext_parquet(
            dataset_config.subset,
            base_directory=dataset_config.base_directory if dataset_config.base_directory else '../Datasets/openwebtext_parquet/',
            engine='pyarrow'
        )
    elif dataset_config.name == 'dolma_v1.6_sample':
        dataset = load_and_concat_dolma_sample(dataset_config.subset)
    elif dataset_config.name == 'pile_uncopyrighted':
        dataset = load_and_concat_pile(dataset_config.subset)
    elif dataset_config.name == 'pile_uncopyrighted_parquet':
        dataset = load_and_concat_pile_parquet(dataset_config.subset)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_config.name}")
    return dataset

    

def load_and_concat_pile_parquet(subset):
    # Define the features
    features = Features({
        "text": Value("string"),
        "meta": Value("string"),
        "id": Value("string")
    })

    # Base directory and subset
    base_directory = '../Datasets/pile-uncopyrighted_parquet/default/partial-train/'

    # Load and concatenate each file in the subset
    all_data = []
    for index in subset:
        file_path = os.path.join(base_directory, f'{index:04d}.parquet')
        df = pd.read_parquet(file_path)
        
        # Process the 'meta' field
        df['meta'] = df['meta'].apply(lambda x: list(x.values()))
        
        # Add the 'id' column
        df['id'] = df.index.map(lambda idx: f"{index}-{idx}")
        
        all_data.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)

    # Convert the DataFrame to a Dataset object
    dataset = Dataset.from_pandas(combined_df)

    return dataset


def load_and_concat_pile(subset):
    def load_jsonl_zst(file_path):
        data = []
        with open(file_path, 'rb') as compressed_file:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(compressed_file) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                for line in text_stream:
                    data.append(json.loads(line))
        return data
    # Define the features
    features = Features({
        "text": Value("string"),
        "meta": Value("string")
    })

    # Base directory and subset
    base_directory = '../Datasets/pile-uncopyrighted/train/'

    # Load and process each file in the subset
    all_data = []
    for index in subset:
        file_path = os.path.join(base_directory, f'{index:02d}.jsonl.zst')
        data = load_jsonl_zst(file_path)
        for idx, item in enumerate(data):
            item['id'] = f"{index}-{idx}"
            item['meta']=item['meta'].values()
        all_data.extend(data)

    # Extract texts and metas
    texts = [d.get('text', '') for d in all_data]
    metas = [d.get('meta', '') for d in all_data]
    ids=[d.get('id', '') for d in all_data]
    
    # Create a Dataset object
    dataset = Dataset.from_dict({
        'text': texts,
        'meta': metas,
        'id':ids,
    })

    return dataset

def load_and_concat_dolma_sample(subset, base_directory = '../Datasets/dolma_v1.6_sample/'):
    def load_json_gz(file_path):
        data = []
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    # Define the features
    features = Features({
        "id": Value("string"),
        "text": Value("string"),
        "added": Value("string"),
        "created": Value("string"),
        "source": Value("string"),
    })

    # Base directory and subset

    # Load and process each file in the subset
    all_data = []
    for index in subset:
        file_path = os.path.join(base_directory, f'v1_5r2_sample-{index:04d}.json.gz')
        data = load_json_gz(file_path)
        for idx, item in enumerate(data):
            item['id'] = f"{index}-{idx}"
        all_data.extend(data)

    # Create a Dataset object
    dataset = Dataset.from_dict({key: [d[key] for d in all_data] for key in features.keys()}, features=features)

    return dataset



def load_and_concat_openwebtext_parquet(subset, base_directory='../Datasets/openwebtext_parquet/', engine='pyarrow'):
    """
    Loads specified Parquet files and concatenates them into a single DataFrame.

    Parameters:
    - subset: list of integers, specifying which Parquet files to load.
    - base_directory: str, the base path where the Parquet files are stored.
    - engine: str, either 'pyarrow' or 'fastparquet', specifying the engine for reading Parquet files.

    Returns:
    - pd.DataFrame: A DataFrame containing the concatenated data from the specified Parquet files.
    """
    dataframes = []  # To store individual DataFrames before concatenation

    for index in subset:
        # Generate the file path from the index
        file_path = os.path.join(base_directory, f'{index:04d}.parquet')
        
        # Check if the file exists before attempting to load
        if os.path.exists(file_path):
            # Load the Parquet file
            df = pd.read_parquet(file_path, engine=engine)
            df['id'] = [f"{index}-{i}" for i in range(len(df))]
            dataframes.append(df)
        else:
            print(f"File not found: {file_path}")
    
    # Concatenate all DataFrames if any have been loaded
    if dataframes:
        return Dataset.from_pandas(pd.concat(dataframes, ignore_index=True))
    else:
        return None  # Return an empty DataFrame if no files were loaded



