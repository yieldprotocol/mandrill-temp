from transformers import Trainer

class MandrillTrainer(Trainer):
    """
    avoid setting label to None: https://github.com/huggingface/transformers/blob/5a4f340df74b42b594aedf60199eea95cdb9bed0/src/transformers/trainer.py#L2703C26-L2703C26
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        outputs = model(**inputs)
        return outputs.loss