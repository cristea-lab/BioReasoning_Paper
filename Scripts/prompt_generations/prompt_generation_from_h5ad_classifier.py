#!/usr/bin/env python3

"""
Usage:
  python create_prompts.py --input <input_file.h5ad> --output <output_file.ndjson>
                           --cell-types-file <cell_types.txt>
                           [--top-genes TOP_GENES] [--all-nonzero]

Description:
  This script reads an h5ad file containing single-cell data and extracts metadata 
  (cell type, tissue, disease, development_stage) and either:
    - The top N expressed genes (by --top-genes), or
    - All non-zero expressed genes (by --all-nonzero)
  for each cell. Using this information, it constructs two prompts (long and short)
  for Large Language Model (LLM) tasks. It then writes the information (including
  the prompts) for each cell into an NDJSON file (each line is a valid JSON object).

  The script also requires a list of possible cell types (one per line) to be passed
  via --cell-types-file. The prompts instruct the model to pick only from this list.

Dependencies:
  - scanpy
  - numpy
  - scipy
  - json
"""

import argparse
import scanpy as sc
import numpy as np
from scipy.sparse import issparse
import json

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM prompts from an h5ad single-cell dataset."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input .h5ad file.")
    parser.add_argument("--output", "-o", required=True, help="Path to output .ndjson file.")
    parser.add_argument(
        "--cell-types-file", "-c", required=True,
        help="Path to the text file containing possible cell types (one per line)."
    )
    parser.add_argument(
        "--top-genes", "-t", type=int, default=100,
        help="Number of top genes to include (default=100). Ignored if --all-nonzero is used."
    )
    parser.add_argument(
        "--all-nonzero", action="store_true",
        help="If set, include all non-zero expressing genes, sorted by descending expression. "
             "This overrides --top-genes."
    )

    args = parser.parse_args()
    input_h5ad = args.input
    output_ndjson = args.output
    cell_types_file = args.cell_types_file

    # Load the list of valid cell types
    with open(cell_types_file, "r") as f:
        # Strip newlines and skip empty lines
        cell_types_list = [line.strip() for line in f if line.strip()]

    # Convert that list into a multiline string for the prompt
    # (Alternatively, you can format this any way you like in the prompt)
    cell_types_str = "\n".join(cell_types_list)

    # Read the .h5ad file using scanpy
    print(f"Reading h5ad file from {input_h5ad}...")
    adata = sc.read_h5ad(input_h5ad)

    # Prepare a list to hold the cell dictionaries
    cell_dicts = []

    # Loop over each cell by index
    print("Constructing cell information and prompts...")
    for i, cell_id in enumerate(adata.obs.index):
        
        # Retrieve the cell's expression vector
        if issparse(adata.X):
            expr = adata.X[i, :].toarray().flatten()
        else:
            expr = adata.X[i, :]

        # Sort gene indices by descending expression
        sorted_indices = np.argsort(expr)[::-1]

        if args.all_nonzero:
            # Include all genes with non-zero expression
            nonzero_mask = expr[sorted_indices] > 0
            top_gene_indices = sorted_indices[nonzero_mask]
        else:
            # Include top N genes (default=100)
            top_gene_indices = sorted_indices[:args.top_genes]

        # Map indices to gene names. 
        # Make sure your adata.var has a column named 'feature_name' or adjust as needed.
        top_genes = adata.var.iloc[top_gene_indices]['feature_name'].tolist()

        # Pull values from adata.obs for this cell
        # Adjust to your actual column names if they differ
        soma_joinid_val = adata.obs.loc[cell_id, 'soma_joinid']
        soma_joinid_val = int(soma_joinid_val)  # convert from np.int64 to int

        # Build a dictionary for this cell
        cell_info = {
            'cell_name': cell_id,
            'soma_joinid': soma_joinid_val,
            'cell_type_ground_truth': str(adata.obs.loc[cell_id, 'cell_type']),
            'tissue': str(adata.obs.loc[cell_id, 'tissue']),
            'disease': str(adata.obs.loc[cell_id, 'disease']),
            'development_stage': str(adata.obs.loc[cell_id, 'development_stage']),
            'genes': top_genes
        }

        # Construct the prompts
        genes_str = " ".join(top_genes)

        # Long prompt
        long_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cell, followed by a list of its genes in descending expression.\n\n"
            f"Tissue: {cell_info['tissue']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Genes: {genes_str}\n\n"
            "Here is a list of all possible cell types you must choose from:\n"
            f"{cell_types_str}\n\n"
            "Please identify what cell type this might be.\n"
            "Please pick the single best matching cell type from this list.\n"
            "At the end of your response, strictly place the final lines in this format:\n\n"
            "Cell type: X\n"
        )

        # Short prompt
        short_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cell, followed by a list of its genes in descending expression.\n\n"
            f"Tissue: {cell_info['tissue']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Genes: {genes_str}\n\n"
            "Here is a list of all possible cell types you must choose from:\n"
            f"{cell_types_str}\n\n"
            "Please identify what cell type this might be.\n"
            "Please pick the single best matching cell type from this list.\n"
            "Keeping your response concise and clear.\n"
            "At the end, use this format:\n\n"
            "Cell type: X\n"
        )

        # Add prompts to the cell info
        cell_info['prompt'] = long_prompt
        cell_info['short_prompt'] = short_prompt

        cell_dicts.append(cell_info)

    # Write out to NDJSON
    print(f"Writing output to {output_ndjson}...")
    with open(output_ndjson, "w") as outfile:
        for obj in cell_dicts:
            outfile.write(json.dumps(obj) + "\n")

    print("Done.")

if __name__ == "__main__":
    main()