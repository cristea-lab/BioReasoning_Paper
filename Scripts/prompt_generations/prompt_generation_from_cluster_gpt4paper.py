#!/usr/bin/env python3

"""
Usage:
  python create_prompts_from_clusters.py --input <clusters.csv> --output <clusters.ndjson>

Description:
  This script reads a CSV file with cluster-level single-cell data. It will:
    - Collect all columns from the CSV (so you can track them in the final NDJSON).
    - Generate per-cluster prompts in the same format as your original cell-level script,
      ensuring downstream code expecting similar fields ('cell_name', 'genes', etc.)
      will still work.
    - Output NDJSON, with one JSON object per line.

Dependencies:
  - python>=3.7
  - pandas
"""

import argparse
import pandas as pd
import json

def main():
    parser = argparse.ArgumentParser(description="Generate LLM prompts from a CSV of cluster-level data.")
    parser.add_argument("--input", "-i", required=True, help="Path to input CSV file.")
    parser.add_argument("--output", "-o", required=True, help="Path to output NDJSON file.")
    args = parser.parse_args()

    input_csv = args.input
    output_ndjson = args.output

    print(f"Reading CSV file from {input_csv} ...")
    df = pd.read_csv(input_csv)

    cluster_dicts = []

    print("Constructing cluster information and prompts...")
    for idx, row in df.iterrows():
        # 1) Start by making a dictionary for *all* CSV columns
        #    (so you keep them for later).
        #    We'll store them as-is, converting to string if needed.
        cell_info = {}
        for col in df.columns:
            # Convert to string to avoid issues with NaNs or numeric columns
            cell_info[col] = str(row[col])

        # 2) Make a "cell_name" (or cluster name) so your downstream code sees it.
        #    If your old pipeline expects 'cell_name', we can just use 'cluster_{idx}' 
        #    or any other unique label from the CSV if you have one.
        cell_info['cell_name'] = f"cluster_{idx}"

        # 3) If your downstream analysis code looks for these fields,
        #    replicate them (or fill with 'NA' if not in your CSV).
        cell_info['cell_type_ground_truth'] = str(row.get("manual_annotation", "NA"))
        cell_info['tissue'] = str(row.get("tissue", "NA"))
        # If your CSV has 'disease' or 'development_stage', pull them in.
        # Otherwise, default them to "NA".
        cell_info['disease'] = str(row.get("disease", "NA"))
        cell_info['development_stage'] = str(row.get("development_stage", "NA"))

        # 4) Parse the top genes from the "marker" column into a list
        marker_str = row.get("marker", "")
        genes = [g.strip() for g in marker_str.split(",") if g.strip()]
        cell_info['genes'] = genes

        # 5) Build the "long" and "short" prompts, similar to your original script
        genes_str = " ".join(genes)

        long_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cluster of cells along with its marker genes:\n"
            f"Tissue: {cell_info['tissue']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Marker genes: {genes_str}\n\n"
            "Please identify what cell type this might be, as granular and accurate as possible.\n"
            "At the end of your response, strictly place the final lines in this format:\n\n"
            "Cell type: X\n"
        )

        short_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cluster of cells along with its marker genes:\n"
            f"Tissue: {cell_info['tissue']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Marker genes: {genes_str}\n\n"
            "Please identify what cell type this might be, as granular and accurate as possible.\n"
            "Keep your response concise and clear.\n"
            "At the end of your response, strictly place the final lines in this format:\n\n"
            "Cell type: X\n"
        )

        cell_info['prompt'] = long_prompt
        cell_info['short_prompt'] = short_prompt

        cluster_dicts.append(cell_info)

    print(f"Writing NDJSON to {output_ndjson} ...")
    with open(output_ndjson, "w") as out:
        for obj in cluster_dicts:
            out.write(json.dumps(obj) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()