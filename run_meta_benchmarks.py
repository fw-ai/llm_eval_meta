import argparse
import asyncio
import pickle
from datasets import load_dataset
from openai import AsyncOpenAI
import openai
from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

# Mapping providers to their clients and models
provider_to_models = {
    "fw": {
        "8b": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "70b": "accounts/fireworks/models/llama-v3p1-70b-instruct",
        "405b": "accounts/fireworks/models/llama-v3p1-405b-instruct",
        "v3-8b": "accounts/fireworks/models/llama-v3-8b-instruct",
        "v3-70b": "accounts/fireworks/models/llama-v3-70b-instruct",
        "v3-8b-fp16": "accounts/fireworks/models/llama-v3-8b-instruct-hf",
        "v3-70b-fp16": "accounts/fireworks/models/llama-v3-70b-instruct-hf",
    },
    "tg": {
        "8b": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "70b": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "405b": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    },
}


# Define the retry strategy
retry_strategy = retry(
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    retry=retry_if_exception_type(Exception),  # Retry on any exception
)


# Define the fetch_responses function with retry strategy
@retry_strategy
async def fetch_responses(
    client, prompt, semaphore, index, provider, model_size, output_dir, max_tokens
):
    output_file = os.path.join(output_dir, f"response_{index}.pkl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping.")
        return

    async with semaphore:
        response = await client.completions.create(
            model=provider_to_models[provider][model_size],
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        if isinstance(response, openai.BadRequestError):
            with open(output_file, "wb") as f:
                pickle.dump("bad_response", f)
        assert isinstance(response, openai.types.completion.Completion)
        # Save response to a file
        with open(output_file, "wb") as f:
            pickle.dump(response, f)


TASK_TO_MAX_TOKENS = {
    "evals__mmlu__details": 1,
    "evals__mmlu__0_shot__cot__details": 1024,
    # Official meta uses 1024, but a small % (.5) of questions are answered correctly after relaxing
    "evals__mmlu_pro__details": 2048,
    "evals__gsm8k__details": 1024,
}


def get_client(provider):
    return {
        "fw": AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1/"),
        "tg": AsyncOpenAI(base_url="https://api.together.xyz/v1"),
    }[provider]
    

# Define the main function
async def main(args):
    ds = load_dataset(
        "meta-llama/Meta-Llama-3.1-405B-Instruct-evals",
        f"Meta-Llama-3.1-405B-Instruct-{args.eval_set}",
    )
    semaphore = asyncio.Semaphore(args.concurrency)  # Limit to 16 concurrent tasks

    if args.num_examples is None:
        args.num_examples = len(ds["latest"]["input_final_prompts"])
    prompts = ds["latest"]["input_final_prompts"][: args.num_examples]

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)

    tasks = []
    # Create the tasks with tqdm progress bar
    max_tokens = TASK_TO_MAX_TOKENS[args.eval_set]
    client = get_client(args.provider)
    for idx, prompt in enumerate(tqdm(prompts, desc="Creating tasks")):
        tasks.append(
            asyncio.create_task(
                fetch_responses(
                    client,
                    f"<|begin_of_|text|>{prompt[0]}",
                    semaphore,
                    idx,
                    args.provider,
                    args.model_size,
                    args.output_dir,
                    max_tokens=max_tokens,
                )
            )
        )

    # Run the tasks with tqdm progress bar
    for future in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing tasks"
    ):
        await future


# Entry point for the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script to run model with specified parameters."
    )
    parser.add_argument(
        "--model-size",
        type=str,
        required=True,
        help="Size of the model (e.g., 8b or 70b)",
    )
    parser.add_argument(
        "--provider", type=str, required=True, help="Provider name (e.g., fw or tg)"
    )
    parser.add_argument("--eval-set", type=str, required=True)
    parser.add_argument(
        "--num-examples", type=int, default=None, help="Number of examples to process"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save responses"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
