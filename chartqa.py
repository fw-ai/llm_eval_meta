import argparse
import asyncio
import base64
from dataclasses import dataclass
from io import BytesIO
import pickle
from typing import Any
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

# Define the retry strategy
retry_strategy = retry(
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    retry=retry_if_exception_type(Exception),  # Retry on any exception
)


# Define the fetch_responses function with retry strategy
@retry_strategy
async def fetch_responses(prompt, semaphore, index, model_name, output_dir, max_tokens):
    output_file = os.path.join(output_dir, f"response_{index}.pkl")
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping.")
        return

    headers = {"Content-Type": "application/json"}
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{prompt.image}"},
                },
                {
                    "type": "text",
                    "text": (
                        f"""{prompt.question}
                        Analyze the image and question carefully, using step-by-step reasoning.
                        First, describe any image provided in detail. Then, present your reasoning. And finally your final answer in this format:
                        Final Answer: <answer>
                        where <answer> follows the following instructions:
                        - <answer> should be a single phrase or number.
                        - <answer> should not paraphrase or reformat the text in the image.
                        - If <answer> is a ratio, it should be a decimal value like 0.25 instead of 1:4.
                        - If the question is a Yes/No question, <answer> should be Yes/No.
                        - If <answer> is a number, it should not contain any units.
                        - If <answer> is a percentage, it should include a % sign.
                        - If <answer> is an entity, it should include the full label from the graph.
                        IMPORTANT: Remember, to end your answer with Final Answer: <answer>."""
                    ),
                },
            ],
        }
    ]

    data = {
        "messages": messages,
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
    }

    url = "http://0.0.0.0:80/v1/chat/completions"

    import httpx

    async with semaphore:
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                response = await client.post(url, headers=headers, json=data)
                response.raise_for_status()
                result = response.json()
                with open(output_file, "wb") as f:
                    print(
                        f"Dumping response to {output_file} with prompt.answer={prompt.answer}"
                    )
                    pickle.dump(
                        (
                            result["choices"][0]["message"]["content"],
                            prompt.answer,
                        ),
                        f,
                    )
            except Exception as e:
                print(f"GOT ERROR", e)
                raise e


def get_client(provider):
    if provider == "fw":
        return AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1/")
    elif provider == "tg":
        return AsyncOpenAI(base_url="https://api.together.xyz/v1")
    elif provider == "local_fw":
        # return AsyncOpenAI(base_url="http://0.0.0.0:80/v1")
        import httpx

        return httpx.AsyncClient(timeout=None)
    else:
        raise ValueError(f"Invalid provider: {provider}")


@dataclass
class Entry:
    type: str
    question: str
    answer: str
    image: Any


def pil_image_to_base64(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")  # Saved as PNG
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_base64


# Define the main function
async def main(args):
    ds = load_dataset("lmms-lab/ChartQA")

    if args.num_examples is None:
        args.num_examples = len(ds["test"])

    prompts = []
    count = 0
    for type, question, answer, image in zip(
        ds["test"]["type"],
        ds["test"]["question"],
        ds["test"]["answer"],
        ds["test"]["image"],
    ):
        prompts.append(Entry(type, question, answer, pil_image_to_base64(image)))
        count += 1
        if count >= args.num_examples:
            break

    os.makedirs(args.output_dir, exist_ok=True)

    semaphore = asyncio.Semaphore(args.concurrency)
    # client = get_client(args.provider)

    tasks = []
    for idx, prompt in enumerate(tqdm(prompts, desc="Creating tasks")):
        tasks.append(
            fetch_responses(
                prompt,
                semaphore,
                idx,
                args.model_name,
                args.output_dir,
                max_tokens=8192,
            )
        )

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
        "--provider",
        type=str,
        required=True,
        help="Provider name (e.g., fw, tg, or local_fw)",
    )
    parser.add_argument(
        "--num-examples", type=int, default=None, help="Number of examples to process"
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Directory to save responses"
    )

    args = parser.parse_args()
    asyncio.run(main(args))
