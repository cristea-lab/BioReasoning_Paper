# ContinueDeepseek.py

import argparse
import json
import os
from openai.types.chat.chat_completion import ChatCompletion

# Import necessary functions from the original script (QueryDeepseek.py)
from QueryDeepseek import (
    load_ndjson,
    save_r1_response, # Re-use for saving
    save_v3_response, # Re-use for saving
    SafeDeepSeek,
    concurrent_query,
    count_failed_query,
    query_remain
)

def continue_main(api_key, input_file, output_file, mode, prompt_type_str, n_worker, report, save_interval_for_continue):
    print("--- Starting Continue Script ---")

    # 0. API and mode setup
    SafeDeepSeek.update_api_key(api_key)
    is_long_prompt = False
    prompt_key_to_use = ''
    current_save_func = None # Will be save_v3_response or save_r1_response

    if mode.lower() == "v3":
        SafeDeepSeek.update_mode('deepseek-chat')
        query_function = SafeDeepSeek.safe_query
        current_save_func = save_v3_response
    elif mode.lower() == "r1":
        SafeDeepSeek.update_mode('deepseek-reasoner') # Default in class
        query_function = SafeDeepSeek.safe_query
        current_save_func = save_r1_response
    else:
        raise ValueError("Invalid mode. Choose either 'v3' or 'r1'.")

    if prompt_type_str.lower() == "long":
        prompt_key_to_use = 'prompt'
        is_long_prompt = True
    elif prompt_type_str.lower() == "short":
        prompt_key_to_use = 'short_prompt'
        is_long_prompt = False
    else:
        raise ValueError("Invalid prompt choice. Choose either 'long' or 'short'.")

    # 1. Load all original cells and ensure 'original_index'
    all_original_cells = load_ndjson(input_file)
    if not all_original_cells:
        print(f"Input file {input_file} is empty or could not be loaded. Exiting.")
        return

    for i, cell_data in enumerate(all_original_cells):
        if 'original_index' not in cell_data:
            cell_data['original_index'] = i # Ensure original_index for consistency
        if 'soma_joinid' not in cell_data: # Critical for diffing
             print(f"Warning: Original cell at index {i} (after potential assignment: {cell_data['original_index']}) is missing 'soma_joinid'. It cannot be reliably tracked for continuation and will likely be skipped.")


    # 2. Load soma_joinids of successfully processed items from the output file
    processed_soma_joinids = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        print(f"Loading previously processed items from {output_file}...")
        try:
            processed_items = load_ndjson(output_file)
            for item in processed_items:
                if 'soma_joinid' in item:
                    processed_soma_joinids.add(item['soma_joinid'])
            print(f"Found {len(processed_soma_joinids)} already processed soma_joinids.")
        except Exception as e:
            print(f"Error loading or parsing {output_file}: {e}. Will proceed assuming only new items need processing if output file is problematic.")
    else:
        print(f"Output file {output_file} does not exist or is empty. Will process all applicable items from input based on its content.")

    # 3. Identify unprocessed cells
    cells_to_retry = []
    query_list_for_retry = []

    for cell in all_original_cells:
        soma_id = cell.get('soma_joinid')
        if not soma_id: # If a cell has no soma_joinid, it can't be diffed; skip.
            continue

        if soma_id not in processed_soma_joinids:
            if prompt_key_to_use not in cell:
                print(f"Warning: Cell with soma_joinid {soma_id} (original_index {cell.get('original_index')}) is missing '{prompt_key_to_use}'. Skipping for retry.")
                continue
            cells_to_retry.append(cell) # cell includes 'original_index'
            query_list_for_retry.append(cell[prompt_key_to_use])

    if not cells_to_retry:
        print("No remaining items to process based on soma_joinid comparison. All done or input error.")
        return

    print(f"Found {len(cells_to_retry)} items to retry based on soma_joinid.")

    # To track items saved periodically *during this continue script's run*
    globally_saved_indices_for_this_continue_run = set()

    # 4. Execute API calls for the identified subset
    print(f"Starting concurrent query for {len(query_list_for_retry)} remaining items with {n_worker} threads ...")
    new_results_for_retry_batch = concurrent_query(
        query_function,
        query_list_for_retry,
        max_workers=n_worker,
        report=report,
        output_file_for_save=output_file,
        original_cells_for_save=cells_to_retry, # Pass the subset being processed
        save_function_for_save=current_save_func,
        long_prompt_flag_for_save=is_long_prompt,
        save_interval_for_save=save_interval_for_continue, # Use the new arg
        globally_saved_indices_for_save=globally_saved_indices_for_this_continue_run
    )

    # 5. Retry failed calls within this batch
    loop_count = 0
    max_retries = 10 # Or make this an arg
    failed_count_in_batch = count_failed_query(new_results_for_retry_batch)
    while failed_count_in_batch > 0 and loop_count < max_retries:
        print(f"Retry attempt {loop_count + 1}/{max_retries} for {failed_count_in_batch} failed queries in this continue batch...")
        query_remain(
            new_results_for_retry_batch,  # The list of results for the current retry batch
            query_list_for_retry,         # The list of queries for this batch
            cells_to_retry,               # The list of cell data for this batch (FIXED)
            query_function,
            max_workers=n_worker,
            report=report,
            output_file_for_save=output_file,
            save_function_for_save=current_save_func,
            long_prompt_flag_for_save=is_long_prompt,
            save_interval_for_save=save_interval_for_continue,
            globally_saved_indices_for_save=globally_saved_indices_for_this_continue_run
        )
        failed_count_in_batch = count_failed_query(new_results_for_retry_batch)
        loop_count += 1

    # 6. Filter newly successful results that weren't saved by periodic saving (if any)
    final_cells_to_append = []
    final_results_to_append = []
    still_failed_soma_joinids_this_run = []

    for i, result_item in enumerate(new_results_for_retry_batch):
        current_cell = cells_to_retry[i] # This cell has 'original_index'
        if isinstance(result_item, ChatCompletion):
            # Check if it was already saved by the periodic mechanism within this run
            if current_cell['original_index'] not in globally_saved_indices_for_this_continue_run:
                final_cells_to_append.append(current_cell)
                final_results_to_append.append(result_item)
        else:
            soma_id = current_cell.get('soma_joinid', f"Unknown_OriginalIndex_{current_cell.get('original_index', 'N/A')}")
            still_failed_soma_joinids_this_run.append(soma_id)
            # Optional: Log error
            # error_msg = result_item.get('error', 'Unknown') if isinstance(result_item, dict) else 'Not ChatCompletion'
            # print(f"Query for soma_joinid: {soma_id} (OriginalIndex: {current_cell.get('original_index')}) ultimately failed in this run. Error: {error_msg}")


    # 7. Append any remaining newly successful results to output_file using imported save functions
    if final_cells_to_append:
        print(f"Appending {len(final_cells_to_append)} remaining successful results from this run to {output_file}...")
        current_save_func(
            output_file,
            final_cells_to_append,
            final_results_to_append,
            long=is_long_prompt,
            append_mode=True # CRITICAL: always append
        )
        print(f"Successfully appended {len(final_cells_to_append)} items.")
    elif len(globally_saved_indices_for_this_continue_run) > 0 and not still_failed_soma_joinids_this_run :
        print("All newly processed items were saved periodically. No final append needed for this run.")
    else:
        print("No new items were successfully processed and ready for final append in this run.")

    total_newly_saved_count = len(globally_saved_indices_for_this_continue_run) + len(final_cells_to_append)
    print(f"Total items newly saved in this ContinueDeepseek run: {total_newly_saved_count}")

    if still_failed_soma_joinids_this_run:
        print(f"The following soma_joinids still failed after this run and were not saved. You might need to run continue.py again:")
        print(still_failed_soma_joinids_this_run)
    elif total_newly_saved_count < len(cells_to_retry):
         print(f"Warning: Not all items targeted in this run ({len(cells_to_retry)}) were successfully processed and saved ({total_newly_saved_count}). Check logs.")
    else:
        print("All items targeted in this continue run appear to have been successfully processed and saved.")

    print("--- Continue Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue processing failed API queries from a previous run, with periodic saving.")
    parser.add_argument("--api_key", type=str, required=True, help="API key for querying Deepseek models")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the ORIGINAL .ndjson input file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the .ndjson file where results are (being) saved (will be appended to)")
    parser.add_argument("--mode", type=str, choices=["v3", "r1"], required=True, help="Mode of API query: 'v3' or 'r1'")
    parser.add_argument("--prompt", type=str, choices=["long", "short"], default='long', help="Type of prompt: 'long' or 'short'")
    parser.add_argument("--n_worker", type=int, default=64, help="Number of worker threads (default: 64)")
    parser.add_argument("--report", type=int, default=100, help="Report interval for concurrent processing (default: 100)")
    parser.add_argument("--save_interval_continue", type=int, default=50,
                        help="Save results every N successful queries DURING THIS CONTINUE RUN. "
                             "0 to disable periodic saving within this script (default: 50).")

    args = parser.parse_args()
    continue_main(args.api_key, args.input_file, args.output_file, args.mode, args.prompt, args.n_worker, args.report, args.save_interval_continue)