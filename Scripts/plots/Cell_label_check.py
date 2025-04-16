'''
This script checks if ground truth cell types are within scTab or scGPT's classifier labels, for splitting cells in after cutoff data.
'''
import os
import json
import pandas as pd
import numpy as np
import subprocess

def load_ndjson(file):
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
def load_json(file):
    with open(file, "r") as f:
        try:
            data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from file: {e}")
            return None

def get_all_children(celltypes, path):
    saved_ndjson = []
    for i,c in enumerate(celltypes):
        saved_ndjson.append({
            'soma_joinid': i,
            'cell_type_ground_truth': c,
            'response': f"Cell type: {c}"
        })
    subprocess.run(f"mkdir {path}", capture_output=True, text=True, shell=True)
    with open(f'{path}/celltypes.ndjson', "w") as outfile:
        for obj in saved_ndjson:
            outfile.write(json.dumps(obj) + "\n")
    subprocess.run(f"cd /cristealab/xiwang/DSR1_Preprint/Scripts/evaluation/ \
                    && ./run_pipelines.sh {path}/celltypes.ndjson {path}", 
        capture_output=True, text=True, shell=True)
    
# get all children for scGPT cell typs
'''with open('/cristealab/rtan/scGPT/DSR1/scGPT_labels.txt', "r") as f:
    # Strip newlines and skip empty lines
    scGPT_labels = [line.strip() for line in f if line.strip()]
get_all_children(scGPT_labels, '/cristealab/xiwang/DSR1_Preprint/Outputs/Celltype_alignment/scGPT_labels')'''

# get all children for scTab cell typs
with open('/cristealab/xiwang/Outputs/llm_sc_benchmark/scTab/subsampled_1k_with_prompts/scTab_unique_cell_types.txt', "r") as f:
    # Strip newlines and skip empty lines
    scTab_labels = [line.strip() for line in f if line.strip()]
get_all_children(scTab_labels, '/cristealab/xiwang/DSR1_Preprint/Outputs/Celltype_alignment/scTab_labels')

# get all children for ground truth cell typs in two datasets
extracted_results_1 = load_ndjson('/cristealab/xiwang/DSR1_Preprint/Outputs/cellxgene_cutoff_8datasets_10k/responses/short_prompt_r1/100genes_scTabClassifier/step1_extracted.ndjson')
extracted_results_2 = load_ndjson('/cristealab/xiwang/DSR1_Preprint/Outputs/cellxgene_cutoff_random_10k/responses/short_prompt_r1/100genes_scTabClassifier/step1_extracted.ndjson')
ground_truth_celltypes = list(set([i['ground_truth'] for i in extracted_results_1]).union(set([i['ground_truth'] for i in extracted_results_2])))
get_all_children(ground_truth_celltypes, '/cristealab/xiwang/DSR1_Preprint/Outputs/Celltype_alignment/groundtruth_labels')

