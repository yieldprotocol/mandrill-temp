from transformers import Trainer
from transformers.trainer_utils import EvalLoopOutput
import evaluate

class MandrillTrainer(Trainer):
    """
    avoid setting label to None: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L2703C26-L2703C26
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        return outputs.loss
    
    def evaluation_loop(self, dataloader, description, prediction_loss_only=False, **kwargs) -> EvalLoopOutput:
        '''
        https://github.com/huggingface/transformers/blob/0a55d9f7376f72ad3ff296d4249840021b03bcc4/src/transformers/trainer_utils.py#L147
        '''
        print('eval stuff goes here...')
        return EvalLoopOutput(predictions=None, label_ids=None, metrics={'fake_metric': 0.0}, num_samples=0)
