from llama.generation import Message, Dialog, B_INST, E_INST, B_SYS, E_SYS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from preprocess.prompts import SYSTEM_PROMPT
import os

HF_CACHE_DIR = os.environ.get('HUGGINGFACE_CACHE_DIR', '/notebooks/.cache/huggingface')
HF_API_TOKEN = os.environ.get('HUGGINGFACE_API_TOKEN', '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')

LLAMA_HF_ID = "meta-llama/Llama-2-7b-chat-hf"
llama_tokenizer = AutoTokenizer.from_pretrained(LLAMA_HF_ID, cache_dir=HF_CACHE_DIR, token=HF_API_TOKEN)
llama_tokenizer.pad_token = llama_tokenizer.eos_token

def llama_dialog2tokens(dialog: Dialog, tokenizer=llama_tokenizer, verbose=False):
    # copied / adapted from https://github.com/facebookresearch/llama/blob/d58f9ae95c299fe6388ee2da2c87fd90cd360d41/llama/generation.py#L284
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_tokens: List[int] = sum(
        [
            tokenizer.encode(
                f"{tokenizer.bos_token}{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {tokenizer.eos_token}",
            )
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    if verbose:
        messages = [
            f"{tokenizer.bos_token} {B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} {tokenizer.eos_token}"
    for prompt, answer in zip(
        dialog[::2],
        dialog[1::2],)]
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{tokenizer.bos_token} {B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
    )
    if verbose:
        messages.append(f"{tokenizer.bos_token} {B_INST} {(dialog[-1]['content']).strip()} {E_INST}")
        display(messages)
    return dialog_tokens

def llama_get_prompt_tokens(jsonl_row, system_message=SYSTEM_PROMPT):
    # TODO: make a dataclass or pydantic type for jsonl_row
    SYSTEM_MESSAGE = Message(role='system', content=SYSTEM_PROMPT)
    dialog = [
        SYSTEM_MESSAGE,
        Message(role='user', content=jsonl_row['instruction']),
    ]
    return llama_dialog2tokens(dialog)

def llama_get_input_with_labels(row, tokenizer=llama_tokenizer):
    prompt_tokens = llama_get_prompt_tokens(row)
    response_tokens = tokenizer(
        f"{row['response']} {tokenizer.eos_token}"
    )['input_ids']
    input_ids = prompt_tokens + response_tokens
    attention_mask = [1] * len(input_ids)
    labels = [-100]*len(prompt_tokens) + response_tokens
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

if __name__ == '__main__':
    sample_dialog: Dialog = [
        {"role": "system", "content": "Welcome to the virtual assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I assist you today?"},
        {"role": "user", "content": "I need some help with Python programming."},
        {"role": "assistant", "content": "Sure, I can help with Python. What do you need assistance with?"},
        {"role": "user", "content": "I'm having trouble with a Python script."},
        {"role": "assistant", "content": "Could you please provide the script or describe the issue you're facing?"},
        {"role": "user", "content": "```python\n"
                                       "def calculate_square(x):\n"
                                       "    return x ** 2\n"
                                       "```"},
        {"role": "assistant", "content": "Thank you for sharing the script. What seems to be the problem with it?"},
        {"role": "user", "content": "I'm getting a 'NameError' for 'x' when I run it."},
        {"role": "assistant", "content": "The 'NameError' indicates that 'x' is not defined. You should provide a value for 'x' when calling the function."},
        {"role": "user", "content": "```python\n"
                                       "def calculate_square(x):\n"
                                       "    x = 5  # Assign a value to x\n"
                                       "    return x ** 2\n"
                                       "```"},
        {"role": "assistant", "content": "Great! Now the 'x' variable is defined. Is there anything else you need help with?"},
        {"role": "user", "content": "No, that's all for now. Thanks for your assistance!"}
    ]

    print(dialog2tokens(sample_dialog, verbose=True))