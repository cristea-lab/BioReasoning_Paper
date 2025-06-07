# Example Command:
# ----------------
#     python QueryDeepseek.py --api_key sk-12345mkdfl --input_file input.ndjson --output_file output.ndjson --mode v3 --prompt short --n_worker 64 --report 100




import argparse
from openai import OpenAI
import json
import concurrent.futures
import requests
import time # For potential future use, not strictly needed for count-based saving
import os # To check if file exists for initial write/append logic (optional)

# input/output functions
def load_ndjson(file):
    '''Read in a .ndjson file, return a list object
    parameters:
        file: path of the .ndjson file
    return: a list
    '''
    output = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    single_cell = json.loads(line)
                    output.append(single_cell)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {e}")
    return output

def save_r1_response(ndjson_path, cells, r1_response, long, append_mode=False):
    '''
    Save response of r1 in .ndjson format
    ndjson_path: str, save path for output, .ndjson name
    cells: original input, for index, soma ID and cell_type_ground_truth
    r1_response: response of r1 queries
    long: if truth, save the long prompt; else short prompt
    append_mode: if True, append to the file, otherwise overwrite
    '''
    r1_response_json = []
    # Ensure cells[0] exists if cells is not empty
    keys_to_keep = []
    if cells:
        keys_to_check = ['soma_joinid', 'cell_type_ground_truth']
        # Include original_index if present, used for mapping back
        if 'original_index' in cells[0]:
             keys_to_check.append('original_index')
        keys_to_keep = [key for key in keys_to_check if key in cells[0]]

    for i in range(len(cells)):
        item = {}
        # Use original_index if available, otherwise fallback to current index i
        item['index'] = cells[i].get('original_index', i)
        for key in keys_to_keep:
            item[key] = cells[i][key]
        item['prompt']= cells[i]['prompt'] if long else cells[i]['short_prompt']
        item['reasoning']=r1_response[i].choices[0].message.reasoning_content
        item['response']= r1_response[i].choices[0].message.content
        item['prompt_token']=r1_response[i].usage.prompt_tokens
        item['reasoning_token']=r1_response[i].usage.completion_tokens_details.reasoning_tokens
        item['completion_token']=r1_response[i].usage.completion_tokens
        r1_response_json.append(item)

    mode = "a" if append_mode else "w"
    with open(ndjson_path, mode) as outfile:
        for obj in r1_response_json:
            outfile.write(json.dumps(obj) + "\n")

def save_v3_response(ndjson_path, cells, v3_response, long, append_mode=False):
    '''
    Save response of v3 in .ndjson format
    ndjson_path: str, save path for output, .ndjson name
    cells: original input, for index and keys to be saved in keys_to_check
    v3_response: response of v3 queries
    long: if truth, save the long prompt; else short prompt
    append_mode: if True, append to the file, otherwise overwrite
    '''
    v3_response_json = []
    keys_to_keep = []
    if cells:
        keys_to_check = ['soma_joinid', 'cell_type_ground_truth']
        if 'original_index' in cells[0]:
             keys_to_check.append('original_index')
        keys_to_keep = [key for key in keys_to_check if key in cells[0]]

    for i in range(len(cells)):
        item = {}
        item['index'] = cells[i].get('original_index', i) # Use original_index
        for key in keys_to_keep:
            item[key] = cells[i][key]
        item['prompt']= cells[i]['prompt'] if long else cells[i]['short_prompt']
        item['response']= v3_response[i].choices[0].message.content
        item['prompt_token']=v3_response[i].usage.prompt_tokens
        item['completion_token']=v3_response[i].usage.completion_tokens
        v3_response_json.append(item)

    mode = "a" if append_mode else "w"
    with open(ndjson_path, mode) as outfile:
        for obj in v3_response_json:
            outfile.write(json.dumps(obj) + "\n")

# call API
API_URL = "https://api.deepseek.com"
from openai.types.chat.chat_completion import ChatCompletion # For type checking

class SafeDeepSeek:
    api_key = 'PlaceHolder'
    base_url = API_URL
    mode = "deepseek-reasoner"

    @classmethod
    def update_api_key(cls, new_api_key):
        cls.api_key = new_api_key

    @classmethod
    def update_mode(cls, new_mode):
        cls.mode = new_mode

    @classmethod
    def safe_query(cls, query):
        client = OpenAI(api_key=cls.api_key, base_url=cls.base_url)
        try:
            response = client.chat.completions.create(
                model=cls.mode,
                messages=[{"role": "user", "content": query}],
                stream=False
            )
            return response
        except Exception as e:
            return {"error": str(e), "query": query}


def concurrent_query(query_function, query_list, max_workers, report=100,
                     # Params for periodic saving
                     output_file_for_save=None,
                     original_cells_for_save=None,
                     save_function_for_save=None,
                     long_prompt_flag_for_save=None,
                     save_interval_for_save=0, # 0 means no periodic saving
                     globally_saved_indices_for_save=None
                     ):
    '''perform concurrent API calls using given query function.
    parameters:
        query_function: a function (like safe_query_deepseek) to query an LLM API
        query_list: a list containing multiple queries for the query function
        max_workers: number of threads for concurrency
        report: the interval to report how many has been processed
        output_file_for_save, original_cells_for_save, save_function_for_save,
        long_prompt_flag_for_save, save_interval_for_save, globally_saved_indices_for_save:
            Parameters passed through for periodic saving.
    return:
        a list that contains results.
    '''
    query_length = len(query_list)
    result_list = ['To be done'] * query_length # Initialize with placeholders
    
    # For periodic saving
    processed_since_last_save = 0
    items_to_save_periodically_cells = []
    items_to_save_periodically_results = []

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map future to original index in the full query_list
            # If query_list is a subset (e.g. from query_remain), this index is relative to that subset
            future_to_original_idx = {
                executor.submit(query_function, query_list[i]): i for i in range(query_length)
            }
            
            total_completed_count = 0
            for future in concurrent.futures.as_completed(future_to_original_idx):
                original_idx_in_current_batch = future_to_original_idx[future] # this is the index within query_list
                result = future.result()
                result_list[original_idx_in_current_batch] = result
                total_completed_count += 1

                # Periodic Save Logic
                if save_interval_for_save > 0 and isinstance(result, ChatCompletion):
                    # This requires original_cells_for_save to map correctly.
                    # If this concurrent_query is called from query_remain, original_idx_in_current_batch
                    # refers to an index in a *subset* of queries. We need the *global* index.
                    # This is handled because query_remain passes subsets of original_cells and query_list.
                    # The `original_index` field in cell data helps map back if needed.
                    
                    # We need to map original_idx_in_current_batch to the index in original_cells_for_save
                    # This assumes original_cells_for_save passed here is the relevant subset if called from query_remain
                    # or the full list if called from main.
                    # We need a robust way to get the *absolute* original index.
                    # Simplification: Assume original_cells_for_save corresponds to query_list
                    
                    # The `globally_saved_indices_for_save` uses the index from the *very original* full `cells` list.
                    # Let's assume `original_cells_for_save[original_idx_in_current_batch]` has an `original_index` field.
                    
                    # This part is tricky if query_list is a sub-list.
                    # The `original_cells_for_save` here should be the full list.
                    # And `original_idx_in_current_batch` needs to be the index in that full list.
                    # This is why we added `original_index` to each cell item when loading/preparing.
                    
                    # For simplicity in this refactor, let's assume for periodic saving,
                    # `original_cells_for_save` corresponds one-to-one with `query_list` being processed here.
                    # And `original_idx_in_current_batch` is the key.
                    # The globally_saved_indices_for_save MUST refer to indices in the *absolute original* `cells` list.
                    # The current `original_idx_in_current_batch` is an index into the `query_list` passed to *this specific* call.
                    # We need a mapping or rely on the `cell['original_index']`.

                    # Let's refine periodic saving to act on the current batch being processed by this function call.
                    # And use the cell's 'original_index' to check against globally_saved_indices_for_save.
                    
                    current_cell_for_saving = original_cells_for_save[original_idx_in_current_batch]
                    global_original_idx = current_cell_for_saving.get('original_index')

                    if global_original_idx is not None and global_original_idx not in globally_saved_indices_for_save:
                        items_to_save_periodically_cells.append(current_cell_for_saving)
                        items_to_save_periodically_results.append(result)
                        processed_since_last_save += 1
                    
                    if processed_since_last_save >= save_interval_for_save and items_to_save_periodically_cells:
                        print(f"Periodically saving {len(items_to_save_periodically_cells)} successful results...")
                        save_function_for_save(
                            output_file_for_save,
                            items_to_save_periodically_cells,
                            items_to_save_periodically_results,
                            long_prompt_flag_for_save,
                            append_mode=True
                        )
                        for cell_saved in items_to_save_periodically_cells:
                            globally_saved_indices_for_save.add(cell_saved['original_index'])
                        
                        items_to_save_periodically_cells = []
                        items_to_save_periodically_results = []
                        processed_since_last_save = 0

                if total_completed_count % report == 0 or total_completed_count == query_length:
                    print(f"Processed {total_completed_count}/{query_length} queries in current batch.")

    except KeyboardInterrupt:
        print('Early stop due to KeyboardInterrupt in concurrent_query.')
        # Fall through to save any pending periodic items
    finally:
        # Final periodic save for any remaining items in this batch
        if save_interval_for_save > 0 and items_to_save_periodically_cells:
            print(f"Saving remaining {len(items_to_save_periodically_cells)} successful results from batch...")
            save_function_for_save(
                output_file_for_save,
                items_to_save_periodically_cells,
                items_to_save_periodically_results,
                long_prompt_flag_for_save,
                append_mode=True
            )
            for cell_saved in items_to_save_periodically_cells:
                 globally_saved_indices_for_save.add(cell_saved['original_index'])
    
    print(f'Batch of {query_length} queries processed.')
    return result_list


def count_failed_query(lst):
    from openai.types.chat.chat_completion import ChatCompletion
    n = sum(isinstance(item, ChatCompletion) for item in lst)
    fail_n = len(lst) - n
    return fail_n

def query_remain(last_run_results, original_queries_full, original_cells_full,
                 query_function, max_workers, report=100,
                 # Periodic saving params
                 output_file_for_save=None, save_function_for_save=None,
                 long_prompt_flag_for_save=None, save_interval_for_save=0,
                 globally_saved_indices_for_save=None):
    failed_n = count_failed_query(last_run_results)
    if failed_n:
        print(f'{failed_n} failed queries from previous run, retrying...')
    else:
        print('No failed queries from previous run.')
        return # No modification needed if no failures

    # Find queries that failed (not ChatCompletion)
    # These indices are for `last_run_results`, `original_queries_full`, `original_cells_full`
    remaining_indices_in_full_list = [
        i for i, res in enumerate(last_run_results) if not isinstance(res, ChatCompletion)
    ]
    
    if not remaining_indices_in_full_list:
        print("All queries successful after check.")
        return

    remaining_queries_subset = [original_queries_full[i] for i in remaining_indices_in_full_list]
    remaining_cells_subset = [original_cells_full[i] for i in remaining_indices_in_full_list] # Pass corresponding cells

    print(f"Retrying {len(remaining_queries_subset)} queries.")
    
    # Run remaining queries
    # Pass the subset of cells for periodic saving context
    remaining_response_subset = concurrent_query(
        query_function, remaining_queries_subset, max_workers, report,
        output_file_for_save=output_file_for_save,
        original_cells_for_save=remaining_cells_subset, # Pass the subset of cells
        save_function_for_save=save_function_for_save,
        long_prompt_flag_for_save=long_prompt_flag_for_save,
        save_interval_for_save=save_interval_for_save,
        globally_saved_indices_for_save=globally_saved_indices_for_save
    )

    # Integrate results back into the main `last_run_results` list
    for i, res_idx_in_full_list in enumerate(remaining_indices_in_full_list):
        last_run_results[res_idx_in_full_list] = remaining_response_subset[i]

    failed_n_after_retry = count_failed_query(last_run_results)
    if failed_n_after_retry:
        print(f'Still have {failed_n_after_retry} failed queries after retry.')
    else:
        print('All queries successfully processed after retries.')
    return


def main(api_key, input_file, output_file, mode, prompt_choice_str, n_worker, report, save_interval):
    SafeDeepSeek.update_api_key(api_key)
    
    cells = load_ndjson(input_file)
    # Add original index to each cell for tracking during periodic saves
    for i, cell in enumerate(cells):
        cell['original_index'] = i
        
    if not cells:
        print("Input file is empty or failed to load.")
        return

    globally_saved_indices = set() # Track indices of cells already written to file

    # Optionally, clear the output file if it exists and we want a fresh start
    # For robust resume, it's often better to just append.
    # If save_interval > 0 and os.path.exists(output_file):
    #    print(f"Warning: Output file {output_file} exists. Will append. Delete it for a fresh run.")
    # Or, to always start fresh (deletes previous progress if script is rerun):
    # if os.path.exists(output_file):
    # os.remove(output_file)
    # print(f"Removed existing output file: {output_file}")


    if mode.lower() == "v3":
        SafeDeepSeek.update_mode('deepseek-chat')
        query_function = SafeDeepSeek.safe_query
        save_func = save_v3_response
    elif mode.lower() == "r1":
        SafeDeepSeek.update_mode('deepseek-reasoner') # Default in class
        query_function = SafeDeepSeek.safe_query
        save_func = save_r1_response
    else:
        raise ValueError("Invalid mode. Choose either 'v3' or 'r1'.")

    prompt_key_to_use = ""
    is_long_prompt = False
    if prompt_choice_str.lower() == "long":
        prompt_key_to_use = 'prompt'
        is_long_prompt = True
    elif prompt_choice_str.lower() == "short":
        prompt_key_to_use = 'short_prompt'
        is_long_prompt = False
    else:
        raise ValueError("Invalid prompt choice. Choose either 'long' or 'short'.")

    query_list = []
    for i, cell in enumerate(cells):
        if prompt_key_to_use not in cell:
            print(f"Warning: Cell at original index {i} is missing '{prompt_key_to_use}'. Skipping.")
            # Add a placeholder or handle error, for now, let's make a dummy query
            query_list.append(f"Error: Missing prompt for cell {i}") 
        else:
            query_list.append(cell[prompt_key_to_use])


    print(f"{len(query_list)} queries, starting concurrent query with {n_worker} threads ...")
    results = concurrent_query(
        query_function, query_list, max_workers=n_worker, report=report,
        output_file_for_save=output_file,
        original_cells_for_save=cells, # Pass the full original cells list
        save_function_for_save=save_func,
        long_prompt_flag_for_save=is_long_prompt,
        save_interval_for_save=save_interval,
        globally_saved_indices_for_save=globally_saved_indices
    )

    loop_count = 0
    max_retries = 10
    while count_failed_query(results) > 0 and loop_count < max_retries:
        print(f"\n--- Retry Loop {loop_count + 1}/{max_retries} ---")
        query_remain(
            results, query_list, cells, # Pass full original query_list and cells
            query_function, max_workers=n_worker, report=report,
            output_file_for_save=output_file,
            save_function_for_save=save_func,
            long_prompt_flag_for_save=is_long_prompt,
            save_interval_for_save=save_interval,
            globally_saved_indices_for_save=globally_saved_indices
        )
        loop_count += 1
    
    if count_failed_query(results) > 0:
        print(f"Warning: After {max_retries} retries, {count_failed_query(results)} queries still failed.")

    # Final save for any successful results not yet saved
    print("\n--- Final Save Check ---")
    successful_cells_to_finalize = []
    successful_results_to_finalize = []
    failed_soma_joinids_at_end = []

    for i, result_item in enumerate(results):
        current_cell = cells[i] # `cells` has 'original_index'
        original_idx = current_cell['original_index']

        if isinstance(result_item, ChatCompletion):
            if original_idx not in globally_saved_indices:
                successful_cells_to_finalize.append(current_cell)
                successful_results_to_finalize.append(result_item)
        else:
            # Log failed items that won't be saved
            if 'soma_joinid' in current_cell:
                failed_soma_joinids_at_end.append(current_cell['soma_joinid'])
            error_message = result_item.get('error', 'Unknown error') if isinstance(result_item, dict) else "Not a ChatCompletion object"
            print(f"Query for cell with original index {original_idx} (soma_joinid: {current_cell.get('soma_joinid', 'N/A')}) ultimately failed. Error: {error_message}")

    if successful_cells_to_finalize:
        print(f"Saving {len(successful_cells_to_finalize)} remaining successful results...")
        save_func(output_file, successful_cells_to_finalize, successful_results_to_finalize,
                  long=is_long_prompt, append_mode=True) # Always append in final save
        for cell_saved in successful_cells_to_finalize: # Update tracker, though not strictly needed at very end
            globally_saved_indices.add(cell_saved['original_index'])
        print(f"Final results appended to {output_file}")
    else:
        print(f"No new successful results to save in the final step to {output_file}.")

    total_saved_count = len(globally_saved_indices)
    print(f"Total {total_saved_count}/{len(cells)} results saved to {output_file}.")
    
    if failed_soma_joinids_at_end:
         print(f"The following soma_joinids ultimately failed and were not saved: {failed_soma_joinids_at_end}")
    elif total_saved_count < len(cells):
        print(f"Warning: Some queries failed and were not saved. Total successful: {total_saved_count}, Total queries: {len(cells)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for processing API queries from a .ndjson file with periodic saving.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for quering Deepseek models")
    parser.add_argument("--input_file", type=str, required=True, help="Input .ndjson filename containing prompts")
    parser.add_argument("--output_file", type=str, required=True, help="Output .ndjson filename to save responses")
    parser.add_argument("--mode", type=str, choices=["v3", "r1"], required=True, help="Mode of API query: 'v3' or 'r1'")
    parser.add_argument("--prompt", type=str, choices=["long", "short"], default='long', help="Mode of prompt: 'long' or 'short'")
    parser.add_argument("--n_worker", type=int, default=64, help="Number of worker threads for concurrent processing (default: 64)")
    parser.add_argument("--report", type=int, default=100, help="Report interval for concurrent processing (default: 100)")
    parser.add_argument("--save_interval", type=int, default=50, help="Save results every N successful queries. 0 to disable periodic saving (default: 50).")
    
    args = parser.parse_args()
    main(args.api_key, args.input_file, args.output_file, args.mode, args.prompt, args.n_worker, args.report, args.save_interval)