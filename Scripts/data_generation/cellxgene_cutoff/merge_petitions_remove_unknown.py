#!/usr/bin/env python3
"""
Filter out cells whose `cell_type` is "unknown" (or any other label you specify)
from an .h5ad file and write the cleaned AnnData object to disk.

Example usage
-------------
    python filter_unknown_celltype.py \
        --input-file  merged_normal.h5ad \
        --output-file merged_normal_filtered.h5ad

Optional arguments let you choose a different column or label:
    python filter_unknown_celltype.py \
        --input-file  data.h5ad \
        --output-file data_no_nan.h5ad \
        --obs-field   cell_type_refined \
        --label       unknown
"""
import argparse
import anndata as ad
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove cells with a given label from an AnnData .h5ad file."
    )
    parser.add_argument("--input-file", required=True,
                        help="Path to the input .h5ad file.")
    parser.add_argument("--output-file", required=True,
                        help="Path where the filtered .h5ad will be saved.")
    parser.add_argument("--obs-field", default="cell_type",
                        help="obs column to inspect (default: cell_type).")
    parser.add_argument("--label", default="unknown",
                        help="Label to remove (default: 'unknown').")
    args = parser.parse_args()

    print(f"[filter] Reading AnnData from: {args.input_file}")
    adata: ad.AnnData = ad.read_h5ad(args.input_file)

    if args.obs_field not in adata.obs:
        raise KeyError(f"obs field '{args.obs_field}' not found in AnnData.")

    # Count how many cells will be dropped
    mask = adata.obs[args.obs_field] == args.label
    n_total = adata.n_obs
    n_drop = int(mask.sum())

    print(f"[filter] Cells in file          : {n_total:,}")
    print(f"[filter] Cells matching '{args.label}': {n_drop:,}")

    if n_drop == 0:
        print("[filter] No matching cells – writing original AnnData unchanged.")
    else:
        print("[filter] Filtering …")
        # tqdm for a consistent look, even though the operation is fast
        for _ in tqdm(range(1), desc="Filtering cells"):
            adata = adata[~mask].copy()

        print(f"[filter] Cells after filtering : {adata.n_obs:,}")

    print(f"[filter] Writing filtered AnnData to: {args.output_file}")
    adata.write_h5ad(args.output_file)
    print("[filter] Done.")


if __name__ == "__main__":
    main()