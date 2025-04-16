"""
Detailed Usage Instructions:

This script (query.py) is a pipeline processor that reads an input .ndjson file containing prompts,
queries a model API concurrently, and saves the responses to an output .ndjson file. The script
supports two modes: 'v3' and 'r1', which determine which API function is used to query the model.
It also implements a retry mechanism that will re-run any failed API calls until all queries return
successful results.

Usage:
------
Run the script from the command line with the following required arguments:

    1. api_key: The api key to query Deepseek models.

    2. input_file: The path to the input .ndjson file containing the prompts.
       - Each line in the file should be a valid JSON object with at least a 'prompt' key.
       - If using the "short prompt" version, the JSON should also include a 'short_prompt' key.
       
    3. output_file: The path where the output .ndjson file with the responses will be saved.
       - The output file will be written in .ndjson format, where each line is a JSON object containing
         the prompt, the response, and token usage details.

    4. mode: The API mode to use. It accepts two values:
       - "v3": Uses the deepseek-chat model via the safe_query_deepseek_v3 function.
       - "r1": Uses the deepseek-reasoner model via the safe_query_deepseek function.
       (The mode also determines the format in which responses are saved.)

    5. prompt: The prompt to use. It accepts two values:
       - "long": Uses 'prompt' in the input prompt.
       - "short": Uses the 'short_prompt'.
       Defualt is 'long'

    6. n_worker: Optional, default 64. An integer representing the number of worker threads to use for concurrent processing.
       - This determines the level of concurrency when making API calls.

    7. report: the report interval. Defualt the function reports every 100 finished API calls.

Example Command:
----------------
    python query.py --api_key sk-3409mkdfl --input_file input.ndjson --output_file output.ndjson --mode v3 --prompt short --n_worker 64 --report 100

This command will:
    - Read prompts from 'input.ndjson'.
    - Use 64 worker threads to concurrently query the API in "v3" mode.
    - Save the resulting responses in 'output.ndjson'.

Processing Flow:
----------------
1. **Loading Data:**
   - The script loads the .ndjson file using the load_ndjson function, which parses each line as JSON.
   
2. **Querying the API:**
   - Based on the selected mode ('v3' or 'r1'), the script selects the appropriate query function.
   - It extracts the 'prompt' field from each JSON object in the input file.
   - The concurrent_query function is used to perform API calls concurrently with the specified number of workers.
   
3. **Handling Failures:**
   - If any API call fails, the script will detect these failures and retry them using the query_remain function.
   - This process is iterative and continues until all API calls are successful.
   
4. **Saving Results:**
   - Once all responses are successfully retrieved, the script saves the responses in .ndjson format.
   - The save function used depends on the mode (save_v3_response for "v3" mode and save_r1_response for "r1" mode).


Append these instructions to the query.py file to help guide users on how to utilize this script effectively.
"""


import argparse
from openai import OpenAI
import json

import concurrent.futures
import requests



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

def save_r1_response(ndjson_path, cells, r1_response, long):
    '''
    Save response of r1 in .ndjson format
    ndjson_path: str, save path for output, .ndjson name
    cells: original input, for index, soma ID and cell_type_ground_truth
    r1_response: response of r1 queries
    long: if truth, save the long prompt; else short prompt
    '''
    r1_response_json = []
    keys_to_check = ['soma_joinid', 'cell_type_ground_truth']
    keys_to_keep = [key for key in keys_to_check if key in cells[0]]
    for i in range(len(cells)):
        item = {}
        item['index'] = i
        for key in keys_to_keep:
            item[key] = cells[i][key]
        item['prompt']= cells[i]['prompt'] if long else cells[i]['short_prompt']
        item['reasoning']=r1_response[i].choices[0].message.reasoning_content
        item['response']= r1_response[i].choices[0].message.content
        item['/cristealab/xiwang/DSR1_Preprint/Outputs/GPT4Paper_Clusters/prompts/']=r1_response[i].usage.prompt_tokens
        item['reasoning_token']=r1_response[i].usage.completion_tokens_details.reasoning_tokens
        item['completion_token']=r1_response[i].usage.completion_tokens
        r1_response_json.append(item)
    with open(ndjson_path, "w") as outfile:
        # 遍历每个 JSON 对象
        for obj in r1_response_json:
            # 将对象转换为 JSON 字符串，并在末尾添加换行符
            outfile.write(json.dumps(obj) + "\n")
def save_v3_response(ndjson_path, cells, v3_response, long):
    '''
    Save response of r1 in .ndjson format
    ndjson_path: str, save path for output, .ndjson name
    cells: original input, for index and keys to be saved in keys_to_check
    v3_response: response of v3 queries
    long: if truth, save the long prompt; else short prompt
    '''
    v3_response_json = []
    keys_to_check = ['soma_joinid', 'cell_type_ground_truth']
    keys_to_keep = [key for key in keys_to_check if key in cells[0]]
    for i in range(len(cells)):
        item = {}
        item['index'] = i
        for key in keys_to_keep:
            item[key] = cells[i][key]
        item['prompt']= cells[i]['prompt'] if long else cells[i]['short_prompt']
        item['response']= v3_response[i].choices[0].message.content
        item['prompt_token']=v3_response[i].usage.prompt_tokens
        item['completion_token']=v3_response[i].usage.completion_tokens
        v3_response_json.append(item)
    with open(ndjson_path, "w") as outfile:
        # 遍历每个 JSON 对象
        for obj in v3_response_json:
            # 将对象转换为 JSON 字符串，并在末尾添加换行符
            outfile.write(json.dumps(obj) + "\n")

# call API
API_URL = "https://api.deepseek.com"
class SafeDeepSeek:
    # 类变量：默认的 API key，可以在运行时修改
    api_key = 'PlaceHolder'
    base_url = API_URL
    mode = "deepseek-reasoner" # default is v3

    @classmethod
    def update_api_key(cls, new_api_key):
        '''更新 API key'''
        cls.api_key = new_api_key

    @classmethod
    def update_mode(cls, new_mode):
        '''更新 API key'''
        cls.mode = new_mode

    @classmethod
    def safe_query(cls, query):
        '''安全调用 deepseek r1 API
        参数：
            query: 用作用户查询的字符串
        返回：
            如果查询失败返回一个 dict，否则返回查询结果
        '''
        # 每次调用时使用当前类变量中的 api_key
        client = OpenAI(api_key=cls.api_key, base_url=cls.base_url)
        try:
            response = client.chat.completions.create(
                model=cls.mode,
                messages=[{"role": "user", "content": query}],
                stream=False
            )
            return response
        except Exception as e:
            # 处理异常情况，返回一个 dict
            return {"error": str(e), "query": query}


def concurrent_query(query_function, query_list, max_workers, report = 100):
    '''perform concurrent API calls using given query function.
    parameters:
        query_function: a function (like safe_query_deepseek) to query an LLM API
        query_list: a list containing multiple queries for the query function
        max_workers: number of threads for concurrency
        report: the interval to report how many has been processed
    return: 
        a list that contains results.
    '''
    query_length = len(query_list)
    result_list = ['To be done' for i in range(query_length)]
    query_index = [i for i in range(query_length)]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query_index = {executor.submit(query_function, query_list[i]): i for i in query_index}
        
            for count, future in enumerate(concurrent.futures.as_completed(future_to_query_index)):
                result = future.result()
                result_list[future_to_query_index[future]] = result
                
                # Print progress every 100 queries
                if count % report == 0:
                    print(f"Processed {count}/{query_length} queries")
    except KeyboardInterrupt:
        print('Early stop due to KeyboardInterrupt')
        return result_list
    print('All processed')
    return result_list

def count_failed_query(lst):
    ''' return number of failed queries in the list, determined by the length of lst and number ChatCompletion as successful calls
    parameters: 
        lst: returned by concurrent_query, a list containing API call results
    return: 
        the number of failed_query
    '''
    from openai.types.chat.chat_completion import ChatCompletion
    n = sum(isinstance(item, ChatCompletion) for item in lst)
    fail_n = len(lst) - n
    return fail_n

def query_remain(last_run_results,last_run_queries, query_function, max_workers, report = 100):
    '''usage: complete results from concurrent_query() if there are failed queries
    parameters:
        last_run_results: a list returned by concurrent_query(), could contain failed queries
        last_run_queries: queries for last_run_results
        query_function, query_list, max_workers, report: for concurrent_query()
    return:
        modify last_run_results directly, return none
    '''
    # check if last_run_results contains failed results
    failed_n = count_failed_query(last_run_results)
    if failed_n: 
        print(f'{failed_n} failed queries')
    else: 
        print('No failed queries')
        return None # stop if no failed queries

    # find queries not finished
    from openai.types.chat.chat_completion import ChatCompletion
    remaining_index = [i for i in range(len(last_run_results)) if type(last_run_results[i]) is not ChatCompletion]
    remaining_queries = [last_run_queries[i] for i in remaining_index]
    # run remaining queries
    remaining_response = concurrent_query(query_function, remaining_queries, max_workers, report)

    # integrate results
    # input is modified
    for i in range(len(remaining_index)):
        last_run_results[remaining_index[i]] = remaining_response[i]
    # check results
    failed_n = count_failed_query(last_run_results)
    if failed_n: 
        print(f'Still have {failed_n} failed queries')
    else: 
        print('All finished')
    return None


def main(api_key, input_file, output_file, mode, prompt, n_worker,report):
    # update api
    SafeDeepSeek.update_api_key(api_key)
    # Load input prompts from ndjson file
    cells = load_ndjson(input_file)
    
    # Determine query function and saving function based on mode
    if mode.lower() == "v3":
        SafeDeepSeek.update_mode('deepseek-chat')
        query_function = SafeDeepSeek.safe_query
        save_func = save_v3_response
    elif mode.lower() == "r1":
        query_function = SafeDeepSeek.safe_query # the default is reasoner
        save_func = save_r1_response
    else:
        raise ValueError("Invalid mode. Choose either 'v3' or 'r1'.")
    # Determine which prompt to use
    if prompt.lower() == "long":
        prompt = 'prompt'
        long = True
    elif prompt.lower() == "short":
        prompt = 'short_prompt'
        long =False
    else:
        raise ValueError("Invalid prompt choice. Choose either 'long' or 'short'.")
    
    # Prepare the list of queries. Assuming each cell contains the key 'prompt'
    query_list = [cell[prompt] for cell in cells]

    # Call API concurrently
    print(f"{len(query_list)} queries, starting concurrent query with {n_worker} threads ...")
    results = concurrent_query(query_function, query_list, max_workers=n_worker, report=report)
    
    # Iteratively re-query for failed API calls until all are successful
    loop_count = 0
    while count_failed_query(results) > 0 and loop_count < 10:
        query_remain(results, query_list, query_function, max_workers=n_worker)
        loop_count += 1

    # Save the results. Here we assume saving the long prompt (set long=True)
    print("Saving results...")
    save_func(output_file, cells, results, long=long)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for processing API queries from a .ndjson file")
    parser.add_argument("--api_key", type=str, required=True, help="API key for quering Deepseek models")
    parser.add_argument("--input_file", type=str, required=True, help="Input .ndjson filename containing prompts")
    parser.add_argument("--output_file", type=str, required=True, help="Output .ndjson filename to save responses")
    parser.add_argument("--mode", type=str, choices=["v3", "r1"], required=True, help="Mode of API query: 'v3' or 'r1'")
    parser.add_argument("--prompt", type=str, choices=["long", "short"], default='long', help="Mode of prompt: 'long' or 'short'")
    parser.add_argument("--n_worker", type=int, default=64, help="Number of worker threads for concurrent processing (default: 64)")
    parser.add_argument("--report", type=int, default=100, help="Report interval for concurrent processing (default: 100)")
    args = parser.parse_args()
    main(args.api_key, args.input_file, args.output_file, args.mode, args.prompt, args.n_worker, args.report)