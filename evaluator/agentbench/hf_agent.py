from src.agent import Agent
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

from transformers import AutoTokenizer


class HuggingFaceChatAgent(Agent):
    def __init__(self, model, model_id, system_prompt, temperature=0, max_new_tokens=32, top_p=0, **kwargs) -> None:
        self.model = model
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        super().__init__(**kwargs)

    def inference(self, history: List[dict]) -> str:
        history = json.loads(json.dumps(history))
        prompt = self.conv2prompt(history)
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        output = self.model.generate(
            input_ids,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
        )
        output = output[0].to("cpu")

        response = self.tokenizer.decode(output[input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def conv2prompt(self, history: List[dict]) -> str:
        for idx in range(len(history)):
            if idx==0:
                prompt  = f"<<SYS>>\\n{self.system_prompt}\\n<</SYS>>\\n\\n{history[idx]['content']}"
            elif idx==1:
                prompt += f"<s>[INST] {prompt.strip()} [/INST] {history[idx]['content'].strip()} </s>"
            elif idx==len(history)-1:
                prompt += f"<s>[INST] {history[idx]['user'].strip()} [/INST]"
            else:
                if history[idx]['role']=='user':
                    prompt += f"<s>[INST] {history[idx]['user'].strip()} [/INST] "
                else:
                    prompt += f"{history[idx]['agent'].strip()} </s>"
        prompt += f"<s>[INST] {history[-1]['user'].strip()} [/INST]"
        return prompt
