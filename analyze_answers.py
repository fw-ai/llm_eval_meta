import argparse
from datasets import load_dataset
import transformers
import pickle
import re
import os
from collections import defaultdict

def get_mmlu_answer(response):
    if response is not None:
        return response.choices[0].text.lstrip().rstrip().upper().replace(".", "")
    return None

def get_mmlu_cot_answer(response):
    pattern = r"The best answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "").replace("*", "")

    pattern = r"the best answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")

    pattern = r"The correct answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")

    pattern = r"the correct answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        return match.group(1).replace(".", "")

def get_answer_gsm8k(response):
    pattern = r"The final answer is (.+)\.?"
    match = re.search(pattern, response.choices[0].text)
    if match:
        s = match.group(1)
        for ok_symbol in ["%", "$"]:
            s = s.replace(ok_symbol, "")
        return s


TASK_TO_ANSWER_EXTRACTOR = {
    "evals__mmlu__details": get_mmlu_answer,
    "evals__mmlu__0_shot__cot__details": get_mmlu_cot_answer,
    "evals__gsm8k__details": get_answer_gsm8k,
    "evals__mmlu_pro__details": get_mmlu_cot_answer,
}


def get_dataset_from_task(task, response_path):
    ds_405b = load_dataset(
        f"meta-llama/Meta-Llama-3.1-405B-Instruct-evals",
        f"Meta-Llama-3.1-405B-Instruct-{task}",
    )
    ds_405b_hash_order = [x[0] for x in ds_405b["latest"]["input_final_prompts_hash"]]
    
    if "70b" in str(response_path) or "8b" in str(response_path):
        if "70" in str(response_path):
            ref_model_ds = load_dataset(
                f"meta-llama/Meta-Llama-3.1-70B-Instruct-evals",
                f"Meta-Llama-3.1-70B-Instruct-{task}",
            )
        else:
            ref_model_ds = load_dataset(
                f"meta-llama/Meta-Llama-3.1-8B-Instruct-evals",
                f"Meta-Llama-3.1-8B-Instruct-{task}",
            )
            
        hash_to_row = {}
        for row in ref_model_ds["latest"]:
            hash_to_row[row["input_final_prompts_hash"][0]] = row
        reordered_rows = []
        for prompt_hash in ds_405b_hash_order:
            reordered_rows.append(hash_to_row[prompt_hash])
        ref_model_ds["latest"] = reordered_rows
        return ref_model_ds
    
    return ds_405b

def main(task, response_path):
    ds = get_dataset_from_task(task, response_path)

    responses = []
    total = len(ds["latest"])
    
    for i in range(0, total):
        response = pickle.load(
            open(os.path.join(response_path, f"response_{i}.pkl"), "rb")
        )
        responses.append(response)
        
    from dataclasses import dataclass
    @dataclass
    class Stats:
        correct: int = 0
        total: int = 0
        meta_correct: int = 0
        
        average : float = None
        
        
    subtask_name_to_stats = defaultdict(lambda: Stats())
    
    for response, ds_row in zip(responses, ds["latest"]):
        model_answer = TASK_TO_ANSWER_EXTRACTOR[task](response)
        
        subtask = ds_row["subtask_name"]
        
        is_eval_correct = model_answer in ds_row["input_correct_responses"]
        if is_eval_correct:
            subtask_name_to_stats[subtask].correct += 1
            
        if ds_row["is_correct"]:
            subtask_name_to_stats[subtask].meta_correct += 1
            
        subtask_name_to_stats[subtask].total += 1
        
    micro_stats = Stats()
    for subtask, stats in subtask_name_to_stats.items():
        stats.average = stats.correct / stats.total
        stats.meta_average = stats.meta_correct / stats.total
        
        micro_stats.correct += stats.correct
        micro_stats.total += stats.total
        micro_stats.meta_correct += stats.meta_correct
        
    micro_stats.average = micro_stats.correct/micro_stats.total
    micro_stats.meta_average = micro_stats.meta_correct/micro_stats.total
    
    import numpy as np
    
    print("Macro average", np.mean([x.average for x in subtask_name_to_stats.values()]))
    print("Meta Macro average", np.mean([x.meta_average for x in subtask_name_to_stats.values()]))
    print("Micro average", micro_stats.average)
    print("Meta Micro average", micro_stats.meta_average)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some tasks and responses.")
    parser.add_argument("--task", type=str, required=True, help="The task to process")
    parser.add_argument(
        "--response-path",
        type=str,
        required=True,
        help="The path to read responses from",
    )

    args = parser.parse_args()
    main(args.task, args.response_path)
