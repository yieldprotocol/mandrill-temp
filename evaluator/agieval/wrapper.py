import os
import torch
from typing import List, Callable
from transformers import AutoTokenizer
from peft import PeftModelForCausalLM

from preprocess.chat import llama_dialog2tokens
from llama.generation import Message
from .AGIEval.src import utils, dataset_loader
from .AGIEval.src import post_process, utils, dataset_loader
from .AGIEval.src import evaluation


run_experiment = True
dataset_dir = "evaluator/agieval/AGIEval/data/v1"
raw_prompt_path = "evaluator/agieval/AGIEval/data/few_shot_prompts.csv"

class HuggingFaceChat():
    def __init__(self, model, model_id, system_prompt, hf_api_token, temperature=0, max_new_tokens=32, top_p=0, batch_size=32, **kwargs) -> None:
        self.model = model
        self.hf_api_token = hf_api_token
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.batch_size = batch_size
        self.system_prompt = system_prompt
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_api_token)
        self.tokenizer.eos_token = "<|im_end|>"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        super().__init__(**kwargs)

    def run_multiple_dataset(self, work_items):
        for item in work_items:
            if os.path.exists(item[1]):
                if len(utils.read_jsonl(item[1])) == item[3]:
                    continue
            self.run_multiple_dataset_batch(item)
            
    def run_multiple_dataset_batch(self, work_item):
        if len(work_item) == 0:
            return
        input_path, output_path, mode, _ = work_item
        assert mode == work_item[2]
        js_list = utils.read_jsonl(input_path)
        content_list = [item["context"] for item in js_list]

        results = []
        for idx in range(0, len(content_list), self.batch_size):
            results += self.query_model_with_retry(context_list=content_list[idx:idx+self.batch_size], setting_name=work_item[2])

        utils.save_jsonl(results, work_item[1])
        assert len(content_list) == len(results)
        
    def query_model_with_retry(self, context_list, setting_name, retry_time=4, results=None):
        if results is None:
            results = self.query_model(context_list, setting_name)
        while retry_time > 0:
            filtered_context_list = []
            for i in range(len(results)):
                if utils.extract_answer(results[i]) == "":
                    filtered_context_list.append(context_list[i])
            if len(filtered_context_list) == 0:
                # print("nothing need to retry")
                break

            filtered_results = self.query_model(filtered_context_list, setting_name)

            p = 0
            for i in range(len(results)):
                if utils.extract_answer(results[i]) == "":
                    results[i] = filtered_results[p]
                    p += 1
            assert p == len(filtered_results)

            retry_succeeded = 0
            for item in filtered_results:
                if utils.extract_answer(item) != "":
                    retry_succeeded += 1
            print("In the retry, {0} samples succeeded, {1} samples failed".format(
                retry_succeeded, len(filtered_results) - retry_succeeded))
            if retry_succeeded <= 3:
                retry_time -= 1
        assert len(results) == len(context_list)
        return results

    def query_model(self, query_list, setting_name='chat') -> str:
        prompt_tokens_list = []
        for query in query_list:
            history = []
            if isinstance(query, str):
                history.append(
                    {"role": "user", "content": query},
                )
            elif isinstance(query, list):
                history += query
            else:
                raise ValueError("Unsupported query: {0}".format(query))
            prompt_tokens = self.conv2tokens(history)
            prompt_tokens_list.append(prompt_tokens)
        params = self.model.config
        bsz = len(prompt_tokens_list)

        min_prompt_len = min(len(t) for t in prompt_tokens_list)
        max_prompt_len = max(len(t) for t in prompt_tokens_list)
        assert max_prompt_len <= params.max_position_embeddings
        total_len = min(params.max_position_embeddings, self.max_new_tokens + max_prompt_len)

        pad_id = self.tokenizer.pad_token_id
        input_ids_batch = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda")
        for k, t in enumerate(prompt_tokens_list):
            input_ids_batch[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        output_batch = self.model.generate(
            input_ids=input_ids_batch,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            top_p=self.top_p,
            eos_token_id= self.tokenizer.eos_token_id,
        )
        output_batch = output_batch.to("cpu")
        response = self.tokenizer.batch_decode(output_batch[:, input_ids_batch.shape[1]:], skip_special_tokens=True)
        response = [{'choices': [{'message': {'content': r}}]} for r in response]
        return response
    
    def conv2tokens(self, history: List[dict]) -> str:
        dialog = [Message(role='system', content=self.system_prompt)]
        dialog += [Message(role=message['role'], content=message['content']) for message in history]
        dialog_tokens = llama_dialog2tokens(dialog)
        return dialog_tokens


def evaluate(model, model_id, system_prompt, hf_api_token,
             temperature=0, max_new_tokens=32, top_p=0, batch_size=2, 
             dataset_name_list=[
                "gaokao-geography",
                "gaokao-history",
                "gaokao-biology",
                "gaokao-chemistry",
                "gaokao-physics",
                "gaokao-mathqa",
                "gaokao-english",
                "sat-math",
                "sat-en", "aqua-rat",
                "lsat-ar", "lsat-lr", "lsat-rc",
                "logiqa-en", "logiqa-zh",
                "gaokao-mathcloze",
                "jec-qa-kd", "jec-qa-ca",
                "math",
                "sat-en-without-passage",], 
             setting_name_list=['few-shot', 'few-shot-CoT', 
                                'zero-shot', 'zero-shot-CoT'], 
             skip_stage_1=False, skip_stage_2=True, skip_stage_3=False, chat_mode=True):

    output_dir = f"./outputs/{model_id.replace('/', '_')}"
    os.makedirs(os.path.join(output_dir, "inputs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "outputs"), exist_ok=True)

    ## Prediction
    model = HuggingFaceChat(model=model, 
                            model_id=model_id, 
                            hf_api_token=hf_api_token,
                            system_prompt=system_prompt, 
                            temperature=temperature, 
                            max_new_tokens=max_new_tokens, 
                            top_p=top_p, 
                            batch_size=batch_size)
    work_items = []
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True, chat_mode=chat_mode)
            # dataset = dataset[:10]
            input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
            utils.save_jsonl(dataset, input_path)
            # dataset = dataset[:10]
            # print(dataset[0]['context'])
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.first_stage.jsonl')

            if 'few-shot' in setting_name:
                work_items.append((input_path, output_path, 'chat' if chat_mode else 'complete', len(dataset)))
            else:
                work_items.append((input_path, first_stage_output_path, 'chat', len(dataset)))

    if not skip_stage_1:
        # model.run_multiple_dataset([item for item in work_items if item[2] == 'complete'])
        model.run_multiple_dataset(([item for item in work_items if item[2] == 'chat']))
        # run_multiple_dataset([item for item in work_items if item[2] == 'complete'])
        # run_multiple_dataset([item for item in work_items if item[2] == 'chat'])

    work_items = []
    for dataset_name in dataset_name_list:
        for setting_name in setting_name_list:
            if 'few-shot' in setting_name:
                continue
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True)
            # dataset = dataset[:10]
            input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl")
            # dataset = dataset[:10]
            # print(dataset[0]['context'])
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.first_stage.jsonl')

            first_stage_results = utils.read_jsonl(first_stage_output_path)
            second_stage_input = dataset_loader.generate_second_stage_input(
                dataset_name, dataset, first_stage_results)
            second_stage_input_path = os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.second_stage.jsonl")
            utils.save_jsonl(second_stage_input, second_stage_input_path)
            work_items.append((second_stage_input_path, output_path, 'chat', len(dataset)))
    if not skip_stage_2:
        model.run_multiple_dataset(work_items)
        # openai_api.default_engine = "chatgpt"
        # run_multiple_dataset(work_items)

    if not skip_stage_3:
        # openai_api.default_engine = "chatgpt"
        wrong_dataset_name_setting_name_list = [
            ("aqua-rat", "few-shot-CoT"),
            ("math", "few-shot"),
            ("math", "few-shot-CoT"),
            ("gaokao-physics", "few-shot-CoT"),
        ]
        for dataset_name, setting_name in wrong_dataset_name_setting_name_list:
            zero_shot_dataset = dataset_loader.load_dataset(
                dataset_name, "zero-shot", dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", verbose=True)
            few_shot_output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.jsonl')
            few_shot_second_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.second_stage.jsonl')
            
    ## Evaluation
    sum_list = [0] * len(setting_name_list)

    print("\t" + "\t".join(setting_name_list))

    for dataset_name in dataset_name_list:
        accuracy_list = []
        for setting_id, setting_name in enumerate(setting_name_list):
            dataset = dataset_loader.load_dataset(
                dataset_name, setting_name, dataset_dir,
                prompt_path=raw_prompt_path, max_tokens=2048,
                end_of_example="<END>\n", chat_mode=chat_mode)
            utils.save_jsonl(dataset, os.path.join(output_dir, "inputs", f"{dataset_name}.{setting_name}.jsonl"))
            output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.jsonl')
            first_stage_output_path = os.path.join(
                output_dir, "outputs", f'predict.{model_id.replace("/", "_")}.{dataset_name}.{setting_name}.first_stage.jsonl')
            second_stage_input_path = os.path.join(
                output_dir, "inputs", f"{dataset_name}.{setting_name}.second_stage.jsonl")

            if not os.path.exists(output_path):
                # print("dataset {0} setting {1} doesn't have results".format(dataset_name, setting_name))
                accuracy_list.append("0")
                continue

            context_list = [item['context'] for item in dataset]

            result_for_human = dataset_loader.load_dataset_as_result_schema(
                dataset_name, dataset_dir
            )

            output_jsons = utils.read_jsonl(output_path)

            if 'zero-shot' in setting_name:
                first_stage_output_jsons = utils.read_jsonl(first_stage_output_path)
                second_stage_input_jsons = utils.read_jsonl(second_stage_input_path)

            for i in range(len(result_for_human)):
                result_for_human[i].model_input = dataset[i]["context"]
                result_for_human[i].model_output = utils.extract_answer(output_jsons[i])
                result_for_human[i].parse_result = post_process.post_process(dataset_name, setting_name, result_for_human[i].model_output)
                result_for_human[i].is_correct = evaluation.evaluate_single_sample(
                    dataset_name, result_for_human[i].parse_result, result_for_human[i].label)
                if 'zero-shot' in setting_name:
                    result_for_human[i].first_stage_output = utils.extract_answer(first_stage_output_jsons[i])
                    result_for_human[i].second_stage_input = second_stage_input_jsons[i]["context"]

            if 'few-shot' in setting_name:
                correct_format = 0
                for i in range(len(result_for_human)):
                    if post_process.try_parse_few_shot_pattern(
                        result_for_human[i].model_output, dataset_name, setting_name):
                        correct_format += 1
                correct_ratio = correct_format / len(result_for_human)
            correct_numer = len([item for item in result_for_human if item.is_correct])
            accuracy = correct_numer / len(result_for_human)
            accuracy_list.append("{0:.2%}".format(accuracy))
            sum_list[setting_id] += accuracy
        print("\t".join([dataset_name] + accuracy_list))
    average_list = []
    for item in sum_list:
        average_list.append("{0:.2%}".format(item/len(dataset_name_list)))
    print("\t".join(["average"] + average_list))