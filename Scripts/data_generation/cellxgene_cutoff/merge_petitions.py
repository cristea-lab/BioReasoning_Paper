#!/usr/bin/env python3
"""
Merge multiple chunked .h5ad files into a single .h5ad.

Example usage:
    python merge_h5ad_partitions.py \
        --input-dir ./normal_unique_h5ad \
        --output-file merged_normal.h5ad

The script will scan for all .h5ad files in --input-dir, read them in,
and concatenate them into a single AnnData object, which is then saved
as --output-file.
"""

import argparse
import os
import glob
from tqdm import tqdm
import anndata as ad

def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple .h5ad partitions into a single .h5ad file."
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing partition_*.h5ad files.")
    parser.add_argument("--output-file", required=True,
                        help="Path to the merged output .h5ad file.")
    args = parser.parse_args()

    # Gather all .h5ad files in the specified directory
    partition_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5ad")))
    if not partition_files:
        raise ValueError(f"No .h5ad files found in {args.input_dir}.")

    print(f"[merge] Found {len(partition_files)} partition .h5ad files to merge.")
    
    # Initialize an empty list to store partial AnnData objects
    merged_adata = None
    total_cells = 0

    # Read and concatenate chunk by chunk
    for pf in tqdm(partition_files, desc="Merging partitions"):
        # Load the chunk
        adata_chunk = ad.read_h5ad(pf)
        chunk_size = adata_chunk.n_obs
        total_cells += chunk_size

        # Concatenate with merged_adata
        if merged_adata is None:
            # First chunk: just take it
            merged_adata = adata_chunk
        else:
            # Merge with existing
            merged_adata = ad.concat([merged_adata, adata_chunk],
                                     axis=0,
                                     join="outer",
                                     merge="unique")
    
    print(f"[merge] Total cells after merging: {total_cells:,}")

    # Write to disk
    print(f"[merge] Writing merged AnnData to: {args.output_file}")
    merged_adata.write_h5ad(args.output_file)
    print("[merge] Done.")

if __name__ == "__main__":
    main()