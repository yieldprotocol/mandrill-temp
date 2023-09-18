from dataclasses import asdict, dataclass, field, fields

@dataclass
class EvaluationArguments:
    """
    https://github.com/huggingface/transformers/blob/bc7ce1808f6c30df87fd9dff871a53ef510ccf77/src/transformers/training_args.py#L159C1-L761C11
    
    EvaluationArguments is the subset of the arguments we use in our example scripts **which relate to the evaluation loop
    itself**.

    Parameters:
        system_prompt (`str`, *optional*):
            The system prompt to use while evaluating
        temperature (`float`, *optional*, defaults to 0.0):
            The model's temperature for evaluating.
        per_device_eval_batch_size (`int`, *optional*, defaults to 16):
            The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation.
        max_new_tokens (`int`, *optional*, defaults to 50):
            Maximum number of tokens to generate in evaluation.
        top_p (`float`, *optional*):
            Parameter for nucleus sampling in decoding.
    """
    
    system_prompt: str = field(
        default="You are a helpful AI assistant",
        metadata={"help": " The system prompt to use while evaluating."},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": ("The model's temperature for evaluating.")},)
    per_device_eval_batch_size: bool = field(default=16, metadata={"help": "The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for evaluation."})
    max_new_tokens: bool = field(default=False, metadata={"help": "Maximum number of tokens to generate in evaluation."})
    top_p: bool = field(default=False, metadata={"help": "Parameter for nucleus sampling in decoding."})