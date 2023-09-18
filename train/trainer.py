from transformers import Trainer, TrainingArguments
from transformers.utils import logging
from transformers.trainer_utils import EvalLoopOutput

from evaluator.agieval import wrapper as wrapper_agieval
from eval_args import EvaluationArguments
# from evaluator.agentbench import wrapper as wrapper_agentbench

logger = logging.get_logger(__name__)

class MandrillTrainer(Trainer):
    """
    avoid setting label to None: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L2703C26-L2703C26
    """
    def __init__(self, *args, **kwargs):
        self.model_id = kwargs.pop('model_id')
        self.eval_args = kwargs.pop('eval_args')
        self.hf_api_token = kwargs.pop('hf_api_token')
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        return outputs.loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, **kwargs) -> EvalLoopOutput:
        '''
        https://github.com/huggingface/transformers/blob/0a55d9f7376f72ad3ff296d4249840021b03bcc4/src/transformers/trainer_utils.py#L147
        '''
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        if 'agieval' in self.eval_args.tasks_list:
            logger.info(f"***** Runnning Evaluation on AGIEval *****")
            wrapper_agieval.evaluate(model=model, model_id=self.model_id, hf_api_token=self.hf_api_token,
                                     system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
                                     max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
                                     batch_size=self.args.per_device_eval_batch_size,)
        if 'agentbench' in self.eval_args.tasks_list:
            logger.info(f"***** Runnning Evaluation on AgentBench *****")
            wrapper_agentbench.evaluate(model=model, model_id=self.model_id, hf_api_token=self.hf_api_token,
                                     system_prompt=self.eval_args.system_prompt, temperature=self.eval_args.temperature, 
                                     max_new_tokens=self.eval_args.max_new_tokens, top_p=self.eval_args.top_p,
                                     batch_size=self.args.per_device_eval_batch_size,)
        return EvalLoopOutput(predictions=None, label_ids=None, metrics={'fake_metric': 0.0}, num_samples=0)
