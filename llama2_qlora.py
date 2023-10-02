import os
from datetime import datetime
import wandb
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, TrainingArguments)

from eval_args import EvaluationArguments
from mandrill_utils.logging_utils import generate_random_string
from preprocess.chat import llama_get_input_with_labels
from preprocess.prompts import SYSTEM_PROMPT
from train.trainer import MandrillTrainer
from train.utils import print_trainable_parameters
from dataloaders.dataloaders import load_datasets
from pprint import PrettyPrinter
import argparse
import yaml

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parse a config file in YAML format.')
    parser.add_argument('file', help='The path to the the YAML file to parse.')
    return parser.parse_args()

args = parse_arguments()

with open(args.file) as file:
    try:
        config = yaml.safe_load(file)
        config['base_run_name'] = config['base_run_name'].format(peft_method=config['peft_method'])
    except yaml.YAMLError as exc:
        print(f'Error in configuration file: {exc}')
        
if HUGGINGFACE_API_TOKEN := os.environ.get('HUGGINGFACE_API_TOKEN'):
    print('retrieved HUGGINGFACE_API_TOKEN from `os.environ`')
else:
    print('HUGGINGFACE_API_TOKEN is not set.')

print('CONFIG'.center(50, '='))
pp = PrettyPrinter(indent=2, sort_dicts=False)
pp.pprint(config)
print('\n'*2)

# TODO: make a dataclass or pydantic model for this
TOY = config['toy']
PEFT_METHOD = config['peft_method']
HF_CACHE_DIR = config.get('huggingface', {}).get('cache_dir')
LR = config.get('lr', 2e-4)
BATCH_SIZE = config.get('batch_size', 1)
N_EPOCHS = config.get('n_epochs', 25)
BASE_RUN_NAME = config['base_run_name']
MODEL_ID = config['model_id']
SAVE_DATA_POINTS = config.get('save_data_points', 2000)
WANDB_PROJECT = config.get('wandb', {}).get('project')
WANDB_TEAM = config.get('wandb', {}).get('team')
GRADIENT_ACCUMULATION_STEPS = config.get('gradient_accumulation_steps', 1)
USE_FP16 = config.get('fp16', False)
SYSTEM_PROMPT = config.get('system_prompt', SYSTEM_PROMPT)
"""
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.set_training.gradient_accumulation_steps
When using gradient accumulation, one step is counted as one step with backward pass. 
Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.
"""

if TOY:
    SAVE_DATA_POINTS = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    WANDB_PROJECT += "-toy"

# model_id = "codellama/CodeLlama-7b-Instruct-hf"
bnb_config = None
if PEFT_METHOD=='qlora':
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID, cache_dir=HF_CACHE_DIR, token=HUGGINGFACE_API_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=HF_CACHE_DIR,
    token=HUGGINGFACE_API_TOKEN,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, LoraConfig(**config.get('lora',{})))
print_trainable_parameters(model)

train_dataset = load_datasets(args.file, system_prompt=SYSTEM_PROMPT)

output_root = "outputs/toy" if TOY else "outputs"
run_name = (
    f"{datetime.today().date()}_{BASE_RUN_NAME}_{generate_random_string(5).lower()}"
)
output_dir = f"{output_root}/{run_name}"
print("output_dir:", output_dir)
save_steps = SAVE_DATA_POINTS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

wandb.init(entity=WANDB_TEAM, project=WANDB_PROJECT, name=run_name, config=config)

print(config.get('eval'))

trainer = MandrillTrainer(
    model=model, 
    model_id=MODEL_ID, 
    hf_api_token=HUGGINGFACE_API_TOKEN,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    args=TrainingArguments(
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=2,
        save_steps=save_steps,
        evaluation_strategy='steps',
        eval_steps=save_steps,
        learning_rate=LR,
        fp16=USE_FP16,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
    ),
    eval_args=EvaluationArguments(
        system_prompt=SYSTEM_PROMPT,
        **config.get('eval', {})
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
wandb.finish()
