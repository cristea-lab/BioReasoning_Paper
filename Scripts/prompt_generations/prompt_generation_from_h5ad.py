#!/usr/bin/env python3

"""
Usage:
  python create_prompts.py --input <input_file.h5ad> --output <output_file.ndjson>
                           [--top-genes TOP_GENES] [--all-nonzero]

Description:
  This script reads an h5ad file containing single-cell data and extracts metadata 
  (cell type, tissue, ethnicity, sex, disease, development_stage) and either:
    - The top N expressed genes (by --top-genes), or
    - All non-zero expressed genes (by --all-nonzero)
  for each cell. Using this information, it constructs two prompts 
  (long and short) for Large Language Model (LLM) tasks. Finally, it writes the 
  information for each cell (including the prompts) into an NDJSON file where each
  line is a valid JSON object.

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

        # Map indices to gene names. Assuming 'feature_name' column in adata.var
        top_genes = adata.var.iloc[top_gene_indices]['feature_name'].tolist()

        # Pull values from adata.obs for this cell
        soma_joinid_val = adata.obs.loc[cell_id, 'soma_joinid']
        soma_joinid_val = int(soma_joinid_val)  # convert from np.int64 to int

        # Build a dictionary for this cell
        cell_info = {
            'cell_name': cell_id,
            'soma_joinid': soma_joinid_val,
            'cell_type_ground_truth': str(adata.obs.loc[cell_id, 'cell_type']),
            'tissue': str(adata.obs.loc[cell_id, 'tissue']),
            # 'self_reported_ethnicity': str(adata.obs.loc[cell_id, 'self_reported_ethnicity']),
            # 'sex': str(adata.obs.loc[cell_id, 'sex']),
            'disease': str(adata.obs.loc[cell_id, 'disease']),
            'development_stage': str(adata.obs.loc[cell_id, 'development_stage']),
            'genes': top_genes
        }

        # Construct the prompts
        genes_str = " ".join(top_genes)
        long_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cell, followed by a list of its genes in descending expression:\n"
            f"Tissue: {cell_info['tissue']}\n"
            # f"Ethnicity: {cell_info['self_reported_ethnicity']}\n"
            # f"Sex: {cell_info['sex']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Genes: {genes_str}\n\n"
            "Please identify what cell type this might be, as granular and accurate as possible.\n"
            "At the end of your response, strictly place the final lines in this format:\n\n"
            "Cell type: X\n"
        )

        short_prompt = (
            "You are an expert in single-cell biology.\n\n"
            "Below is metadata for one cell, followed by a list of its genes in descending expression:\n"
            f"Tissue: {cell_info['tissue']}\n"
            # f"Ethnicity: {cell_info['self_reported_ethnicity']}\n"
            # f"Sex: {cell_info['sex']}\n"
            f"Disease: {cell_info['disease']}\n"
            f"Development stage: {cell_info['development_stage']}\n"
            f"Genes: {genes_str}\n\n"
            "Please identify what cell type this might be, as granular and accurate as possible.\n"
            "Keep your response concise and clear.\n"
            "At the end of your response, strictly place the final lines in this format:\n\n"
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