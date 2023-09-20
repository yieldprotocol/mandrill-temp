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
    def __init__(self, model, model_id, system_prompt, hf_api_token, temperature=0, max_new_tokens=32, top_p=0, batch_size, **kwargs) -> None:
        self.model = model
        self.hf_api_token = hf_api_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_token)

        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        history_list = json.loads(json.dumps(history))
        prompt_tokens_list = []
        for history in history_list:
            prompt_tokens = self.conv2tokens(history)
            prompt_tokens_list.append(prompt_tokens)
        params = self.model.config
        bsz = len(prompt_tokens_list)

        min_prompt_len = min(len(t) for t in prompt_tokens_list)
        max_prompt_len = max(len(t) for t in prompt_tokens_list)
        assert max_prompt_len <= params.max_position_embeddings
        total_len = min(params.max_position_embeddings, self.max_new_tokens + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        input_ids = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens_list):
            input_ids[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        output = []
        for idx in range(0, len(input_ids), self.batch_size):
            output_batch = self.model.generate(
                input_ids=input_ids[idx: idx+self.batch_size],
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
            )
            output.append(output_batch)
        output = torch.tensor(output).to("cpu")

        response = self.tokenizer.decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def conv2tokens(self, history: List[dict]) -> str:
        dialog = [Message(role='system', content=self.system_prompt)]
        dialog += [Message(role=message['role'], content=message['content']) for message in history]
        dialog_tokens = llama_dialog2tokens(dialog)
        return dialog_tokens
