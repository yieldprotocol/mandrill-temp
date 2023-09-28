import jsonlines
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
from preprocess.chat import llama_tokenizer, llama_dialog2tokens
from preprocess.prompts import SYSTEM_PROMPT
from llama.generation import Dialog

@dataclass(frozen=True, kw_only=True)
class Chat(ABC):
    id: Optional[str] = None
    system_prompt: Optional[str] = None
    dataset_name: Optional[str] = None
    system_prompt: Optional[str] = SYSTEM_PROMPT

    @abstractmethod
    def to_llama_prompt(self) -> Dialog:
        pass
    
    @abstractmethod
    def to_llama_target(self) -> Dialog:
        '''
        returns: singleton list of format 
        [ {'role': 'assistant', 'content': <content>} ]
        '''
        pass
    
    def to_llama_dialog(self) -> Dialog:
        # return singleton list containing final assistant response
        return self.to_llama_prompt() + self.to_llama_target()
    
    def to_llama_input_with_labels(self):
        prompt_tokens = llama_dialog2tokens(self.to_llama_prompt())
        response_tokens = llama_tokenizer(
            f"{self.to_llama_target()[0]['content']} {llama_tokenizer.eos_token}"
        )['input_ids']
        input_ids = prompt_tokens + response_tokens
        attention_mask = [1] * len(input_ids)
        labels = [-100]*len(prompt_tokens) + response_tokens
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
    

@dataclass(frozen=True, kw_only=True)
class Alpaca(Chat):
    instruction: str
    input: str
    output: Optional[str]
    text: str
    
    def to_llama_prompt(self) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': self.instruction + self.input},
        ]
    
    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.output}
        ]
    
@dataclass(frozen=True, kw_only=True)
class OpenChat(Chat):
    user: str
    assistant: str
    formatted: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.user},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.assistant}
        ]
    
@dataclass(frozen=True, kw_only=True)
class PromptResponse(Chat):
    prompt: str
    response: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.prompt},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.response}
        ]
            
@dataclass(frozen=True, kw_only=True)
class QuestionResponse(Chat):
    question: str
    response: str
    
    def to_llama_prompt(self, system_prompt=SYSTEM_PROMPT) -> Dialog:
        return [
            {'role': 'system', 'content': self.system_prompt or system_prompt},
            {'role': 'user', 'content': self.question},
        ]

    def to_llama_target(self) -> Dialog:
        return [
            {'role': 'assistant', 'content': self.response}
        ]
    
@dataclass(frozen=True, kw_only=True)
class Llama2Turn:
    instruction: Dialog
    response: Dialog
    dataset_name: Optional[str]
    
NAME2CLS = {
    'prompt-response': PromptResponse,
    'question-response': QuestionResponse,
    'alpaca': Alpaca,
    'openchat': OpenChat,
}