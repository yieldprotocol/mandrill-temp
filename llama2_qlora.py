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
from train.trainer import MandrillTrainer
from train.utils import print_trainable_parameters


HUGGINGFACE_API_TOKEN = "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"

TOY = True
BATCH_SIZE = 2
BASE_RUN_NAME = "llama2-7b-qlora"
SAVE_DATA_POINTS = 2000
HF_CACHE_DIR = "/notebooks/.cache/huggingface"
WANDB_PROJECT = "mandrill"
WANDB_TEAM = "yieldinc"
"""
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments.set_training.gradient_accumulation_steps
When using gradient accumulation, one step is counted as one step with backward pass. 
Therefore, logging, evaluation, save will be conducted every gradient_accumulation_steps * xxx_step training examples.
"""
GRADIENT_ACCUMULATION_STEPS = 4

if TOY:
    SAVE_DATA_POINTS = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    WANDB_PROJECT += "-toy"

model_id = "meta-llama/Llama-2-7b-chat-hf"
# model_id = "codellama/CodeLlama-7b-Instruct-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id, cache_dir=HF_CACHE_DIR, token=HUGGINGFACE_API_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"": 0},
    cache_dir=HF_CACHE_DIR,
    token=HUGGINGFACE_API_TOKEN,
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)
print_trainable_parameters(model)

data = load_dataset("json", data_files="data/instructions.jsonl")
data = data.map(llama_get_input_with_labels)

output_root = "outputs/toy" if TOY else "outputs"
run_name = (
    f"{datetime.today().date()}_{BASE_RUN_NAME}_{generate_random_string(5).lower()}"
)
output_dir = f"{output_root}/{run_name}"
print("output_dir:", output_dir)
save_steps = SAVE_DATA_POINTS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)

wandb.init(entity=WANDB_TEAM, project=WANDB_PROJECT, name=run_name)

trainer = MandrillTrainer(
    model=model, 
    model_id=model_id, 
    hf_api_token=HUGGINGFACE_API_TOKEN,
    train_dataset=data["train"],
    eval_dataset=data["train"],
    args=TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        warmup_steps=2,
        save_steps=save_steps,
        evaluation_strategy='steps',
        eval_steps=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_8bit",
    ),
    eval_args=EvaluationArguments(
        system_prompt="You are a helpful AI assistant",
        tasks_list=["agieval"],
        temperature=0.2,
        max_new_tokens=32,
        top_p=0.2,
    ),
    data_collator=transformers.DataCollatorForSeq2Seq(tokenizer),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
wandb.finish()
