#!/usr/bin/env python3
"""
Detailed Usage Instructions:

This script (query.py) is a pipeline processor that reads an input .ndjson file containing prompts,
queries an OpenAI model (via the new openai Python client) concurrently, and saves the results
to an output .ndjson file. It also provides:

1. **Optional Single-Test** (`--test_random`):
   - Randomly picks one prompt to confirm the API works, then asks if you wish to proceed.

2. **Concurrency**:
   - Uses ThreadPoolExecutor to send your prompts in parallel.

3. **Robust Retry Logic**:
   - Uses the built-in OpenAI `max_retries` for short-term backoff on 429 or 5xx errors.
   - Additionally, we implement a higher-level manual **exponential backoff** if some calls still fail
     (e.g., if concurrency is high). We'll wait 2^attempt seconds between each re-try batch.

Usage Example:
--------------
    python query.py \\
        --api_key sk-XXXXX \\
        --model gpt-4o \\
        --input_file my_prompts.ndjson \\
        --output_file results.ndjson \\
        --prompt long \\
        --n_worker 16 \\
        --report 10 \\
        --max_retries 5 \\
        --test_random

If any queries fail with rate-limit or other transient errors, we do a short exponential backoff
and then re-run only those failures, up to `--max_retries` times. If they still fail, you'll see
an `"error"` field in the final `.ndjson`.
"""

import argparse
import json
import random
import time
import concurrent.futures
from openai import OpenAI, APIError

###############################################################################
# I/O FUNCTIONS
###############################################################################
def load_ndjson(path: str):
    """
    Load a .ndjson file into a list of Python dicts.
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                items.append(data)
            except json.JSONDecodeError as e:
                print(f"JSON parse error on line {line_number}: {e}")
    return items

def save_to_ndjson(path: str, prompts: list, responses: list, use_long_prompt: bool):
    """
    Save the results to a .ndjson file.
    """
    chosen_field = "prompt" if use_long_prompt else "short_prompt"
    data_to_write = []

    for i, resp in enumerate(responses):
        record = {}
        record["index"] = i
        # Copy all fields from the input except the raw prompt fields (optional).
        # If you want to keep them, you can just do record.update(prompts[i]).
        for k, v in prompts[i].items():
            if k not in ("prompt", "short_prompt"):
                record[k] = v

        # Which prompt was used
        record["prompt"] = prompts[i].get(chosen_field, "")

        if isinstance(resp, dict) and "error" in resp:
            # This indicates a failure or an exception
            record["error"] = resp["error"]
            record["output_text"] = None
        else:
            # On success, we can retrieve output_text from the new API
            record["output_text"] = resp.output_text

        data_to_write.append(record)

    with open(path, "w", encoding="utf-8") as outfile:
        for item in data_to_write:
            outfile.write(json.dumps(item) + "\n")

###############################################################################
# OPENAI CALL
###############################################################################
def safe_query_openai(client: OpenAI, model: str, instructions: str, user_input: str):
    """
    Safely call the new OpenAI Responses API.
    If it fails, return a dict with "error" in it.
    """
    try:
        resp = client.responses.create(
            model=model,
            instructions=instructions,
            input=user_input,
        )
        return resp
    except APIError as e:
        return {"error": f"OpenAI API error: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

###############################################################################
# SINGLE-TEST FUNCTION
###############################################################################
def test_random_prompt(
    client: OpenAI,
    model: str,
    instructions: str,
    records: list,
    use_long_prompt: bool
):
    """
    Randomly pick one record from 'records', call the new OpenAI API for testing,
    and print out the result or any error.
    """
    if not records:
        print("No records loaded; cannot do single-test.")
        return

    idx = random.randint(0, len(records) - 1)
    chosen_field = "prompt" if use_long_prompt else "short_prompt"
    user_prompt = records[idx].get(chosen_field, "")
    print(f"\n[TEST] Randomly picked record index={idx}, prompt:\n{user_prompt}\n")

    # Call the new API
    response = safe_query_openai(
        client=client,
        model=model,
        instructions=instructions,
        user_input=user_prompt
    )

    if isinstance(response, dict) and "error" in response:
        print("[TEST] API call failed with error:")
        print(response["error"])
    else:
        print("[TEST] API call success! output_text:")
        print(response.output_text)

###############################################################################
# CONCURRENCY
###############################################################################
def run_concurrent_queries(
    client: OpenAI,
    model: str,
    prompt_list: list,
    instructions: str,
    n_worker: int,
    report: int
):
    """
    Perform concurrent calls to safe_query_openai.
    Returns a list of responses in the same order as prompt_list.
    """
    n_prompts = len(prompt_list)
    results = [None] * n_prompts

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_worker) as executor:
        future_map = {
            executor.submit(safe_query_openai, client, model, instructions, prompt_list[i]): i
            for i in range(n_prompts)
        }
        done_count = 0
        for future in concurrent.futures.as_completed(future_map):
            i = future_map[future]
            try:
                results[i] = future.result()
            except Exception as e:
                results[i] = {"error": f"Unhandled exception: {str(e)}"}
            done_count += 1
            if (done_count % report) == 0:
                print(f"Processed {done_count}/{n_prompts} queries")
    print("All processed.")
    return results

def count_failures(responses: list):
    """Count how many items in 'responses' are dicts with an 'error' key."""
    return sum(isinstance(r, dict) and "error" in r for r in responses)

def retry_failures(
    responses: list,
    prompt_list: list,
    client: OpenAI,
    model: str,
    instructions: str,
    n_worker: int,
    report: int
):
    """
    Retries any responses that contain an error, updating 'responses' in place.
    """
    failed_indices = [
        i for i, r in enumerate(responses)
        if isinstance(r, dict) and "error" in r
    ]
    if not failed_indices:
        return

    print(f"{len(failed_indices)} queries failed; retrying...")

    # Build a sub-list of the prompts that failed
    to_retry_prompts = [prompt_list[i] for i in failed_indices]
    new_results = run_concurrent_queries(client, model, to_retry_prompts, instructions, n_worker, report)
    # Insert them back in
    for i, nr in zip(failed_indices, new_results):
        responses[i] = nr

###############################################################################
# MAIN SCRIPT
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Query OpenAI's new Responses API with concurrency, plus optional single-prompt test, "
            "and robust retry/backoff handling for rate limits."
        )
    )
    parser.add_argument("--api_key", type=str, required=True, help="OpenAI API key.")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g. 'gpt-4o', 'gpt-4o-mini', etc.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to your .ndjson file of prompts.")
    parser.add_argument("--output_file", type=str, required=True, help="Where to save the .ndjson results.")
    parser.add_argument("--prompt", type=str, choices=["long", "short"], default="long",
                        help="Which field to use ('long' -> 'prompt', 'short' -> 'short_prompt'). Default='long'.")
    parser.add_argument("--n_worker", type=int, default=64, help="Number of threads for concurrency. Default=64.")
    parser.add_argument("--report", type=int, default=100, help="Reporting interval. Default=100.")
    parser.add_argument("--instructions", type=str, default="You are a helpful AI assistant.",
                        help="System-level instructions for the new Responses API.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max attempts to fix failed queries. Default=5.")
    parser.add_argument("--test_random", action="store_true",
                        help="Randomly pick one prompt for a single test call before processing the whole file.")
    args = parser.parse_args()

    # A) Construct the OpenAI client with built-in short-term retry for 429/500 errors
    #    The default is 2 tries, so let's bump it to e.g. 3 or 4 if desired:
    client = OpenAI(
        api_key=args.api_key,
        max_retries=3,  # built-in short backoff for rate-limit or 5xx
    )

    # B) Load .ndjson
    records = load_ndjson(args.input_file)
    if not records:
        print("No valid prompts found in the input file. Exiting.")
        return

    # C) Optionally test one random prompt
    use_long_prompt = (args.prompt == "long")
    if args.test_random:
        test_random_prompt(
            client=client,
            model=args.model,
            instructions=args.instructions,
            records=records,
            use_long_prompt=use_long_prompt
        )
        user_input = input("\nContinue with full concurrency run? (y/n) ")
        if not user_input.lower().startswith("y"):
            print("Exiting without processing all prompts.")
            return
        print("Proceeding with full concurrency...")

    # D) Extract the user prompt from each record
    field_name = "prompt" if use_long_prompt else "short_prompt"
    prompt_list = [r.get(field_name, "") for r in records]

    # E) Perform concurrency
    print(f"Submitting {len(prompt_list)} prompts to model='{args.model}' with {args.n_worker} workers.")
    results = run_concurrent_queries(
        client=client,
        model=args.model,
        prompt_list=prompt_list,
        instructions=args.instructions,
        n_worker=args.n_worker,
        report=args.report
    )

    # F) Higher-Level Retry Loop with Exponential Backoff
    attempt = 0
    while count_failures(results) > 0 and attempt < args.max_retries:
        # Sleep with exponential backoff before next attempt
        backoff_seconds = 2 ** attempt
        print(f"Rate-limit/backoff: sleeping {backoff_seconds} seconds before retry {attempt + 1} ...")
        time.sleep(backoff_seconds)

        retry_failures(
            responses=results,
            prompt_list=prompt_list,
            client=client,
            model=args.model,
            instructions=args.instructions,
            n_worker=args.n_worker,
            report=args.report
        )
        attempt += 1

    # G) Save final results
    print(f"Saving final results to {args.output_file}")
    save_to_ndjson(args.output_file, records, results, use_long_prompt)
    print("Done.")

if __name__ == "__main__":
    main()