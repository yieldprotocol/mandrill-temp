from .AgentBench.src.agent import Agent
import os
import json
import sys
import time
import re
import math
import random
import datetime
import argparse
import requests
from typing import List, Callable
import dataclasses
from copy import deepcopy

import torch
from transformers import AutoTokenizer
from llama.generation import Message

from preprocess.chat import llama_dialog2tokens


class HuggingFaceChatAgent(Agent):
    def __init__(self, model, model_id, system_prompt, hf_api_token, temperature=0, max_new_tokens=32, top_p=0, batch_size=2, **kwargs) -> None:
        self.model = model
        self.hf_api_token = hf_api_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_token,  padding_side='left')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        history = json.loads(json.dumps(history))
        for h in history:
            if h['role'] == 'agent':
                h['role'] = 'assistant'
        prompt_tokens = self.conv2tokens(history)
        params = self.model.config

        prompt_len = len(prompt_tokens)
        assert prompt_len <= params.max_position_embeddings
        total_len = min(params.max_position_embeddings, self.max_new_tokens + prompt_len)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((1, total_len), pad_id, dtype=torch.long, device="cuda")
        input_ids[0, : len(prompt_tokens)] = torch.tensor(prompt_tokens, dtype=torch.long, device="cuda")
       
        output = self.model.generate(
            input_ids=input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
        )
        output = output.to("cpu")

        response = self.tokenizer.decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def conv2tokens(self, history: List[dict]) -> str:
        dialog = [Message(role=message['role'], content=message['content']) for message in history]
        dialog_tokens = llama_dialog2tokens(dialog)
        return dialog_tokens
