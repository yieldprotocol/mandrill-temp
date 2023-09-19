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
import yaml
from copy import copy
from typing import List, Dict, Any, Optional, Union, Tuple, Callable, Type, TypeVar

import time
import importlib
import argparse

from os.path import join, isdir, isfile, relpath
from glob import glob

from .AgentBench.src import Task, Agent
from .create_assignment import InstanceFactory, Assigment, deep_merge
from .AgentBench.src.utils import ColorMessage


def parse_args_to_assignment(task_name, agent, output=None, workers=None) -> Assigment:
    if not output:
        output = "outputs/" + task_name
    
    if task_name in ["os_interaction", "dbbench", "knowledgegraph", "card_game", 
                   "lateralthinking_puzzle", "mind2web", "alfworld", "webshop"]:
        task_config = f"evaluator/agentbench/{task_name}_dev.yaml"
    else:
        ValueError("unsupported task")
    try:
        task_config = json.loads(task_config)
        if isinstance(task_config, str):
            raise Exception()
    except:
        with open(task_config, "r", encoding='utf-8') as f:
            task_config = yaml.safe_load(f)

    agent_config = "evaluator/agentbench/hfchat.yaml"
    try:
        agent_config = json.loads(agent_config)
        if isinstance(agent_config, str):
            raise Exception()
    except:
        with open(agent_config, "r", encoding='utf-8') as f:
            agent_config = yaml.safe_load(f)
    print(task_config)
    print(agent_config)
    if "workers" not in task_config["parameters"]:
        task_config["parameters"]["workers"] = 1
    if workers:
        task_config["parameters"]["workers"] = workers
    agent_config['parameters']['model'] = agent
    
    return Assigment(agent=InstanceFactory(**agent_config), task=InstanceFactory(**task_config), output=output)

# register a signal handler to release task
def get_single_handler(task):
    def signal_handler(sig, frame):
        print(ColorMessage.red(f"Received signal {sig}, exiting ..."))
        if isinstance(task, Task):
            task.release()
        sys.exit(0)
    return signal_handler

def create_cfg_yaml(model_id, hf_api_token, system_prompt, 
             temperature, max_new_tokens, top_p):
    cfg_dict = \
    {
        'module': 'evaluator.agentbench.hf_agent.HuggingFaceChatAgent',
        'parameters': {
            'model_id': str(model_id),
            'system_prompt': str(system_prompt),
            'hf_api_token': str(hf_api_token),
            'max_new_tokens': max_new_tokens,
            'temperature': temperature,
            'top_p': top_p
        }
    }
    with open('evaluator/agentbench/hfchat.yaml', 'w') as file:
        yaml.safe_dump(cfg_dict, file, default_flow_style=False)

def evaluate(agent, model_id, hf_api_token, system_prompt, task_name='knowledgegraph', 
             temperature=0.2, max_new_tokens=128, top_p=0.2, batch_size=16, output=None, workers=None):
    # create a yaml config
    create_cfg_yaml(model_id, hf_api_token, system_prompt, 
                 temperature, max_new_tokens, top_p)

    assignment = parse_args_to_assignment(task_name, agent, output, workers)
    # create_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if not assignment.output:
        assignment.output = "outputs" + "/" + task_name
        
    os.makedirs(assignment.output, exist_ok=True)
    
    print(ColorMessage.cyan("[Evaluation] Loading Agent ..."))
    agent = assignment.agent.create()
    print(ColorMessage.cyan("[Evaluation] Successfully loaded Agent."))
    print(ColorMessage.cyan("[Evaluation] Loading Task ..."))
    task = assignment.task.create()
    task.output_root_dir = assignment.output
    print(ColorMessage.cyan("[Evaluation] Successfully loaded Task."))
    config_path = os.path.join(assignment.output, "config.json")
                 
    assignment_ = copy(assignment)
    assignment_.agent.parameters['model'] = assignment_.agent.parameters['model'].__class__.__name__
    with open(config_path, "w", encoding='utf-8') as f:
        f.write(json.dumps(assignment_.to_json(), indent=4, ensure_ascii=False))

    start = time.time()
    # register a signal handler to release task
    import signal
    signal.signal(signal.SIGTERM, get_single_handler(task))
    task.evaluate(agent)
    task.release()
    del task
    print(ColorMessage.cyan(f"[Evaluation] Finish in {time.time() - start:.1f}s"))