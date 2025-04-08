import asyncio
import os
import pickle

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os


def parse_answer(fw_answer):
    fw_answer = fw_answer.split("FINAL ANSWER:")[1]
    return fw_answer.strip()


# Define the retry strategy
retry_strategy = retry(
    stop=stop_after_attempt(5),  # Stop after 5 attempts
    wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff
    retry=retry_if_exception_type(Exception),  # Retry on any exception
)


@retry_strategy
async def fetch_responses(
    client,
    pred,
    ref,
    semaphore,
):
    # Construct the prompt for ChatGPT
    prompt = f"""
    Are these two answers equivalent? 
    
    They don't need to be an exact match, just close enough is correct.
    Consider percentages (%) equivalent to their decimal form (e.g., 50% = 0.5).

    Please consider things correct even if it missing a unit. For example '13 years' is equivalent to '13'.

    Please consider things correct even if one is missing a %. For example '30%' is equivalent to '30'. Which is also equivalent to '0.3'.

     1.6 million t. should match 1.6

    Only reply with Yes or No.

    Answer 1: {pred}
    Answer 2: {ref}
    """

    async with semaphore:
        response = await client.chat.completions.create(
            model="accounts/fireworks/models/deepseek-v3-0324",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        return response.choices[0].message.content.strip()


async def main():
    references = []
    predictions = []
    for i in range(2500):
        file_path = f"/home/yingliu/llm_eval_meta/fw_maverick_chartqa/response_{i}.pkl"
        if not os.path.exists(file_path):
            print(f"{i=}: file not found")
            continue
        ans = pickle.load(open(file_path, "rb"))
        fw_raw_response = ans[0]
        answer = ans[1]
        references.append(answer)
        predictions.append(parse_answer(fw_raw_response))

    from openai import AsyncOpenAI

    tasks = []
    from tqdm import tqdm

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key="OvN1JEAYD7pAdN20djrZPZnxs0Ap7QxLlXVzGnAnHSW2FK1Q",
        timeout=None,
    )
    semaphore = asyncio.Semaphore(64)
    for pred, ref in tqdm(
        zip(predictions, references),
        total=len(predictions),
        desc="Checking equivalence",
    ):
        tasks.append(asyncio.create_task(fetch_responses(client, pred, ref, semaphore)))

    for future in tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Processing tasks"
    ):
        await future

    correct = 0
    incorrect = 0
    for idx, task in enumerate(tasks):
        if task.result() == "Yes":
            correct += 1
        elif task.result() == "No":
            incorrect += 1
            print(f"Incorrect: {idx}")
            print("Prediction:", predictions[idx])
            print("Reference:", references[idx])
        else:
            print(f"Error: {task.result()}", idx)
    print(f"% CORRECT = {correct / len(tasks)}")


if __name__ == "__main__":
    asyncio.run(main())
