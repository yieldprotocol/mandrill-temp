import asyncio
import json
import os
import random
import re
from termcolor import colored 


async def generate(instructor, **kwargs):
    """Generator for bbh_hard chain-of-thought training data."""
    async for item in generate_bbh_hard(
        instructor, "bbh_hard", filter_response=False, **kwargs
    ):
        yield item


async def generate_bbh_hard(
    instructor,
    category,
    filter_response=True,
    only_instructions=False,
    template_kwargs={},
    **kwargs,
):
    """Generator for simple instruction response tasks (e.g. roleplay, wordgames)."""
    config = instructor.instructors.get(category)
    if not config:
        return
    target_count = config.get("count")
    if target_count is None:
        target_count = instructor.default_count
    target_count = int(target_count)
    if not target_count:
        return

    # Load the prompt template.
    path = config.get("prompt_path", f"{category}.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
    with open(path) as infile:
        template = infile.read()

    # Response prompt template (optional).
    response_prompt = None
    path = config.get("response_prompt_path", f"{category}_response.txt")
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", path)
        if os.path.exists(path):
            with open(path) as infile:
                response_prompt = infile.read()

    # API params, overriding defaults with this instructor's config.
    api_params = {**instructor.api_params, **config.get("api_params", {})}

    # Min similarity score.
    min_score = config.get("min_docsearch_score")
    if min_score is None:
        min_score = instructor.min_docsearch_score
    min_score = float(min_score)

    # Load the topics.
    topics = instructor.get_instructor_topics(config)
    topic_index = random.randint(0, len(topics) - 1)

    # Generate the instruction/response pairs until we reach the target count.
    batch_size = config.get("batch_size")
    if batch_size is None:
        batch_size = instructor.default_batch_size
    batch_size = int(batch_size)
    if category not in instructor.instructor_counts:
        instructor.instructor_counts[category] = 0
    language = config.get("language") or instructor.language
    flesch = config.get("flesch") or instructor.default_flesch
    while instructor.instructor_counts[category] < target_count:
        format_args = {"language": language, "flesch": flesch}
        if "{batch_size}" in template:
            format_args["batch_size"] = batch_size
        for key, val in template_kwargs.items():
            format_args[key] = val(instructor)
        if "{topic_avoidance}" in template:
            format_args["topic_avoidance"] = instructor.topic_avoidance
        if "{topics}" in template:
            # Inject the topics to use for this batch.
            current_topics = []
            for _ in range(batch_size):
                current_topics.append(topics[topic_index])
                topic_index += 1
                if topic_index >= len(topics):
                    topic_index = 0
            topics_str = "\n".join(
                [
                    f" * TSK {idx + 1} must be related to topic: {json.dumps(topic)}"
                    for idx, topic in enumerate(current_topics)
                ]
            )
            format_args["topics"] = topics_str
        if "{example1}" in template:
            current_script_path = os.path.realpath(__file__)
            data_folder_path = '/'.join(current_script_path.split('\\')[:-2]) + "/seed_data/bbh_hard"
            file_name = random.choice(os.listdir(data_folder_path))
            with open(f"{data_folder_path}/{file_name}", 'r') as infile:
                data = json.load(infile)
            random.shuffle(data['outputs'])
            format_args["example1"] = '\n\nQ:'.join([data['outputs'][0]['input'].split('\n\nQ:')[0], data['outputs'][0]['input'].split('\n\nQ:')[-1]]).replace('\nA:', '')
            format_args["example2"] = '\n\nQ:'.join([data['outputs'][1]['input'].split('\n\nQ:')[0], data['outputs'][1]['input'].split('\n\nQ:')[-1]]).replace('\nA:', '')
            format_args["example3"] = '\n\nQ:'.join([data['outputs'][2]['input'].split('\n\nQ:')[0], data['outputs'][2]['input'].split('\n\nQ:')[-1]]).replace('\nA:', '')
            format_args["example4"] = '\n\nQ:'.join([data['outputs'][3]['input'].split('\n\nQ:')[0], data['outputs'][3]['input'].split('\n\nQ:')[-1]]).replace('\nA:', '')


        # Get a batch of instructions.
        prompt = template.format(**format_args)
        print(colored('prompt: '+prompt, 'blue'))
        response = await instructor.generate_response(
            prompt, filter_response=filter_response, **api_params
        )
        print(colored('response: '+response, 'yellow'))
        if not response:
            continue

        # Parse instructions and generate responses.
        futures = []
        instructions = []
        for instruction in re.findall(
            r"(?:^|\n)TSK \d+\.\s*(.*?)(?:$|(?=\nTSK \d+\.\s*))", response, re.DOTALL
        ):  
            if not instruction.strip() or await instructor.is_too_similar(
                instruction, min_score=min_score
            ):
                continue
            instructions.append(instruction)
            if only_instructions:
                yield {"instruction": instruction}
            else:
                full_prompt = instruction
                if response_prompt:
                    full_prompt = response_prompt.format(
                        language=language, instruction=instruction, flesch=flesch
                    )
                futures.append(
                    instructor.generate_response(
                        full_prompt,
                        messages=kwargs.get("messages", []),
                        filter_response=filter_response,
                        **api_params,
                    )
                )
        if not futures:
            continue
        responses = await asyncio.gather(*futures)
        for idx in range(len(responses)):
            response = responses[idx]
            if not response or not response.strip():
                continue
            yield {
                "instruction": instructions[idx].strip(),
                "response": response.strip(),
                "category": category,
            }
            if instructor.instructor_counts[category] >= target_count:
                break
