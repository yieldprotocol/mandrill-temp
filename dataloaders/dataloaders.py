import yaml
from typing import TypedDict, Optional
import datasets
import os
import random
random.seed(42)
from dataloaders.data_formats import NAME2CLS
from preprocess.prompts import SYSTEM_PROMPT


HUGGINGFACE_CACHE_DIR = os.environ.get('HUGGINGFACE_CACHE_DIR', '/notebooks/.cache/huggingface')

class DatasetMeta(TypedDict):
    name: str
    path: str
    path_type: str
    format: Optional[str]
    train_slice: Optional[str]

def parse_datasets(yaml_path) -> DatasetMeta:
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    
    datasets = data.get('datasets', [])
    parsed_datasets = []
    
    for dataset in datasets:
        name = dataset.get('name')
        local_path = dataset.get('local_path')
        hf_path = dataset.get('hf_path')
        data_format = dataset.get('format')
        train_slice = dataset.get('train_slice')
        sample_size = dataset.get('sample_size')
        
        if local_path and hf_path:
            raise ValueError(f"Dataset {name} has both local_path and hf_path defined. Only one is allowed.")
        
        if not (local_path or hf_path):
            raise ValueError(f"Dataset {name} must have either local_path or hf_path defined.")
            
        if local_path:
            assert local_path.endswith('.jsonl'), "Local path must be in .jsonl format"
        
        parsed_dataset = {
            'name': name,
            'path': local_path or hf_path,
            'path_type': 'local' if local_path else 'huggingface',
            'format': data_format,
        }
        
        if train_slice:
            parsed_dataset['train_slice'] = train_slice
        
        if sample_size:
            parsed_dataset['sample_size'] = sample_size
        
        parsed_datasets.append(parsed_dataset)
    
    return parsed_datasets

def to_llama(row, format: str, system_prompt:str=None):
    dialog_cls = NAME2CLS[format]
    row['system_prompt'] = system_prompt
    dialog = dialog_cls(**row)
    return dialog.to_llama_input_with_labels()

def load_dataset(dataset_meta: DatasetMeta, split='train', format_fn=to_llama, system_prompt=SYSTEM_PROMPT):
    print(dataset_meta)
    kwargs = {
        'cache_dir': HUGGINGFACE_CACHE_DIR,
    }
    slice = dataset_meta.get(f'{split}_slice')
    if slice:
        slice = f"[{slice}]"
        kwargs['split'] = f"{split}{slice}",
    if dataset_meta['path_type'] == 'local':
        dataset = datasets.load_dataset("json", data_files=dataset_meta['path'], **kwargs)
    
    elif dataset_meta['path_type'] == 'huggingface':
        dataset = datasets.load_dataset(dataset_meta['path'], **kwargs)
    else:
        raise ValueError(f'Could not load dataset {dataset_meta}')
        
    dataset = dataset[split]
        
    if sample_size := dataset_meta.get('sample_size'):
        random_indices = random.sample(range(len(dataset)), sample_size)
        dataset = dataset.select(random_indices) # TODO: add seeds
    
    dataset = dataset.map(lambda row: to_llama(row, format=dataset_meta['format'], system_prompt=system_prompt))
    return dataset

def load_datasets(yaml_path, system_prompt=SYSTEM_PROMPT, do_concat=True):
    parsed_datasets = parse_datasets(yaml_path)
    
    if not do_concat:
        return {dataset['name']: load_dataset(dataset, system_prompt=system_prompt) for dataset in parsed_datasets}
    
    else:
        loaded_datasets = [load_dataset(dataset, system_prompt=system_prompt) for dataset in parsed_datasets]
        return datasets.concatenate_datasets(loaded_datasets)
