#!/usr/bin/env python3

"""
Example Usage:
    python ols_label_normalize_ndjson.py \
        --input_ndjson input_data.ndjson \
        --output_ndjson output_data.ndjson

Optional args:
    --parallel         Enable parallel requests to EBI OLS (default is single-threaded).
    --max_workers N    Number of parallel workers if --parallel is used (default=10).

Script Overview:
    1) Reads an NDJSON (newline-delimited JSON) file. Each line must have keys:
         "ground_truth", "predicted_cell_type"
       plus any other fields (e.g. "soma_joinid").
    2) Collects all unique labels from those keys.
    3) Queries EBI OLS exactly once per unique label (reducing requests
       if labels are repeated across many lines).
    4) Writes a new NDJSON file. For each input line, it appends:
         ground_truth_label_clean, ground_truth_id
         predicted_label_clean,    predicted_id

Example NDJSON input (one line):
  {"soma_joinid": 14055810, "ground_truth": "cerebellar granule cell", "predicted_cell_type": "Cerebellar granule neuron"}

Example output (shortened):
  {
    "soma_joinid": 14055810,
    "ground_truth": "cerebellar granule cell",
    "predicted_cell_type": "Cerebellar granule neuron",
    "ground_truth_label_clean": "cerebellar granule cell",
    "ground_truth_id": "CL:0000120",
    "predicted_label_clean": "cerebellar granule cell",
    "predicted_id": "CL:0000120"
  }

(IDs and labels will depend on OLS' actual top hit.)

Important Considerations:
    - Large parallel requests may risk rate-limiting. Adjust --max_workers as needed.
    - If the same cell-type labels appear repeatedly, this approach saves time by
      querying OLS only once per unique label.
"""

import argparse
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def search_ols_for_cl_label(free_text: str, timeout=15):
    """
    Query EBI OLS for 'free_text' in the Cell Ontology (CL).
    Returns a tuple (official_label, cl_id) or ("", "") if no match.
    """
    base_url = "https://www.ebi.ac.uk/ols/api/search"
    params = {
        "q": free_text,
        "ontology": "cl",
        "type": "class"
    }

    if not free_text or free_text.strip() == "":
        return ("", "")

    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        if resp.status_code != 200:
            return ("", "")

        data = resp.json()
        hits = data.get("response", {}).get("docs", [])
        if not hits:
            return ("", "")

        # Grab the top hit
        best_hit = hits[0]
        ols_label = best_hit.get("label", "")
        ols_obo_id = best_hit.get("obo_id", "")
        # Must start with CL:
        if ols_obo_id and ols_obo_id.startswith("CL:"):
            return (ols_label, ols_obo_id)
        else:
            return ("", "")
    except requests.exceptions.RequestException:
        # covers timeouts, connection errors, etc.
        return ("", "")

def lookup_labels_serial(labels):
    """
    Return a dict {label: (clean_label, cl_id)} 
    by looking up each label in a single-threaded manner.
    """
    results = {}
    for lbl in labels:
        results[lbl] = search_ols_for_cl_label(lbl)
    return results

def lookup_labels_parallel(labels, max_workers=10):
    """
    Return a dict {label: (clean_label, cl_id)} 
    using concurrent requests to speed up lookups.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_label = {executor.submit(search_ols_for_cl_label, lbl): lbl for lbl in labels}
        for future in as_completed(future_to_label):
            lbl = future_to_label[future]
            try:
                results[lbl] = future.result()
            except Exception:
                results[lbl] = ("", "")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_ndjson", required=True,
        help="Path to input NDJSON file containing lines with 'ground_truth' and 'predicted_cell_type'.")
    parser.add_argument("--output_ndjson", required=True,
        help="Path to output NDJSON file with appended OLS fields.")
    parser.add_argument("--parallel", action="store_true",
        help="Use parallel requests to speed up OLS lookups.")
    parser.add_argument("--max_workers", type=int, default=10,
        help="Number of parallel workers if --parallel is used. Default=10.")
    args = parser.parse_args()

    # 1) Read input lines and collect unique labels
    input_records = []
    unique_labels = set()

    with open(args.input_ndjson, "r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            input_records.append(record)

            gt_label = record.get("ground_truth", "").strip()
            pr_label = record.get("predicted_cell_type", "").strip()
            if gt_label:
                unique_labels.add(gt_label)
            if pr_label:
                unique_labels.add(pr_label)

    # 2) Look up labels (serially or in parallel)
    if args.parallel:
        print(f"Using parallel requests with up to {args.max_workers} workers...")
        label_to_info = lookup_labels_parallel(unique_labels, max_workers=args.max_workers)
    else:
        print("Using single-threaded (serial) requests...")
        label_to_info = lookup_labels_serial(unique_labels)

    # 3) Write output NDJSON with new fields
    with open(args.output_ndjson, "w", encoding="utf-8") as fout:
        for record in input_records:
            gt_label = record.get("ground_truth", "").strip()
            pr_label = record.get("predicted_cell_type", "").strip()

            # Retrieve OLS info
            gt_clean, gt_id = label_to_info.get(gt_label, ("", ""))
            pr_clean, pr_id = label_to_info.get(pr_label, ("", ""))

            # Append new fields
            record["ground_truth_label_clean"] = gt_clean
            record["ground_truth_id"] = gt_id
            record["predicted_label_clean"] = pr_clean
            record["predicted_id"] = pr_id

            fout.write(json.dumps(record))
            fout.write("\n")

    print(f"Done! Output written to {args.output_ndjson}")

if __name__ == "__main__":
    main()