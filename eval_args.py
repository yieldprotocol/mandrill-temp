from typing import List
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
        tasks_list (`List[str]`, *optional*):
            The tasks to evaluate the model on.
        temperature (`float`, *optional*, defaults to 0.2):
            The model's temperature for evaluating.
        max_new_tokens (`int`, *optional*, defaults to 50):
            Maximum number of tokens to generate in evaluation.
        top_p (`float`, *optional*):
            Parameter for nucleus sampling in decoding.
        agieval_datasets (`List[str]`, *optional*, defaults to `["math"]`): 
            AGIEval datasets to evaluate on. 
    """
    
    system_prompt: str = field(
        default="You are a helpful AI assistant",
        metadata={"help": " The system prompt to use while evaluating."},
    )
    tasks_list: List[str] = field(
        default_factory=lambda: ["agieval"],
        metadata={"help": " The system prompt to use while evaluating."},
    )
    agieval_datasets: List[str] = field(default_factory=lambda: ['math'], metadata={"help": "List of AGIEval datasets to evaluate on"})
    temperature: float = field(
        default=0.2,
        metadata={"help": ("The model's temperature for evaluating.")},)
    max_new_tokens: bool = field(default=False, metadata={"help": "Maximum number of tokens to generate in evaluation."})
    top_p: float = field(default=0.2, metadata={"help": "Parameter for nucleus sampling in decoding."})
    