import os
import sys
import shutil
import torch
import torch.nn as nn
import wandb

from transformers import Trainer, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import EvalLoopOutput
from peft import PeftModelForCausalLM
from typing import Any, Dict, Union
from contextlib import contextmanager

from eval_args import EvaluationArguments

logger = logging.get_logger(__name__)

class MandrillTrainer(Trainer):
    """
    avoid setting label to None: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L2703C26-L2703C26
    """
    def __init__(self, *args, **kwargs):
        self.model_id = kwargs.pop('model_id')
        self.eval_args = kwargs.pop('eval_args')
        self.hf_api_token = kwargs.pop('hf_api_token')
        self.oom_count = 0
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        with self.handle_oom():
            outputs = model(**inputs)
            return outputs.loss
        return torch.tensor(0.0, requires_grad=True)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        with self.handle_oom():
            return super().training_step(model, inputs)
        return torch.tensor(0.0, requires_grad=True)
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, **kwargs) -> EvalLoopOutput:
        '''
        https://github.com/huggingface/transformers/blob/0a55d9f7376f72ad3ff296d4249840021b03bcc4/src/transformers/trainer_utils.py#L147
        '''
        
        if dataloader is not None:
            super().evaluation_loop(dataloader, description, prediction_loss_only, **kwargs)
        
        # if isinstance(self.model, PeftModelForCausalLM): self.model = self.model.merge_and_unload()
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        if 'agieval' in self.eval_args.tasks_list:
            print(f"{'/'.join(os.path.dirname(__file__).split('/')[:-1])}/evaluator/agieval/AGIEval")
            sys.path.append(f"{'/'.join(os.path.dirname(__file__).split('/')[:-1])}/evaluator/agieval/AGIEval")
            from evaluator.agieval import wrapper as wrapper_agieval
            sys.path.pop()
            logger.info(f"***** Running Evaluation on AGIEval *****")
            wrapper_agieval.evaluate(model=model, model_id=self.model_id, hf_api_token=self.hf_api_token,
                                     system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
                                     max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
                                     batch_size=self.args.per_device_eval_batch_size, dataset_name_list=self.eval_args.agieval_datasets)
        if 'agentbench' in self.eval_args.tasks_list:
            shutil.copy('evaluator/agentbench/AgentBench/create_assignment.py', 'evaluator/agentbench/create_assignment.py')
            sys.path.append(f"{'/'.join(os.path.dirname(__file__).split('/')[:-1])}/evaluator/agentbench/AgentBench")
            from evaluator.agentbench import wrapper as wrapper_agentbench
            sys.path.pop()
            os.remove('evaluator/agentbench/create_assignment.py')
            logger.info(f"***** Running Evaluation on AgentBench *****")
            wrapper_agentbench.evaluate(agent=model, task_name='knowledgegraph', workers=30, model_id=self.model_id, hf_api_token=self.hf_api_token,
                                     system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
                                     max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
                                     batch_size=self.args.per_device_eval_batch_size,)
            # wrapper_agentbench.evaluate(agent=model, task_name='lateralthinkingpuzzle', workers=30, model_id=self.model_id, hf_api_token=self.hf_api_token,
            #                          system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
            #                          max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
            #                          batch_size=self.args.per_device_eval_batch_size,)
            # wrapper_agentbench.evaluate(agent=model, task_name='dbbench', workers=5, model_id=self.model_id, hf_api_token=self.hf_api_token,
            #                          system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
            #                          max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
            #                          batch_size=self.args.per_device_eval_batch_size,)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics={'fake_metric': 0.0}, num_samples=0)

    @contextmanager
    def handle_oom(self):
        # https://github.com/facebookresearch/fairseq/blob/50a671f78d0c8de0392f924180db72ac9b41b801/fairseq/trainer.py#L188
        try:
            yield
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("WARNING: Catching Out of Memory Error")
                torch.cuda.empty_cache()  
                self.oom_count += 1  
                wandb.log({'oom_count': self.oom_count})
            else:
                raise e

