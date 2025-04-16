#!/usr/bin/env python3
"""
Stage 1: Randomly sample 10,000 cells across many partition_*.h5ad files.
Removes the 'with ... as ...:' usage that caused AttributeError.

Example usage:
    python subsampled_from_partitions.py \
        --partitions-dir /path/to/h5ad_partitions \
        --n-sample 10000 \
        --output-file /path/to/sample_10k.h5ad \
        --random-seed 42
"""

import os
import argparse
import scanpy as sc
import numpy as np
import warnings

def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample N cells across multiple partition_*.h5ad files (two-pass)."
    )
    parser.add_argument("--partitions-dir", required=True,
                        help="Directory with partition_*.h5ad files.")
    parser.add_argument("--n-sample", type=int, default=10000,
                        help="Number of cells to sample total. Default=10000.")
    parser.add_argument("--output-file", required=True,
                        help="Path to write the final .h5ad of sampled cells.")
    parser.add_argument("--random-seed", type=int, default=None,
                        help="Random seed for reproducibility (default=None).")
    args = parser.parse_args()

    partitions_dir = args.partitions_dir
    n_sample = args.n_sample
    output_file = args.output_file
    rand_seed = args.random_seed

    # (A) Set random seed if provided
    if rand_seed is not None:
        np.random.seed(rand_seed)
        print(f"[INFO] Setting random seed = {rand_seed}")

    # 1) Gather partition filenames
    partition_files = sorted(
        f for f in os.listdir(partitions_dir)
        if f.startswith("partition_") and f.endswith(".h5ad")
    )
    if not partition_files:
        raise FileNotFoundError(f"No partition_*.h5ad files found in {partitions_dir}")

    print(f"[1/2] Found {len(partition_files)} partitions in {partitions_dir}. Gathering counts...")

    # First pass: gather partition sizes (n_obs in each)
    partition_cell_counts = []
    total_cells = 0

    for fname in partition_files:
        path = os.path.join(partitions_dir, fname)
        # We'll open in "backed='r'" mode so we don't load everything
        ad = sc.read_h5ad(path, backed="r")
        n_obs = ad.n_obs
        partition_cell_counts.append(n_obs)
        total_cells += n_obs

        # Close the backing file to avoid locking / resource usage
        ad.file.close()
        del ad  # remove from memory references

    print(f"[1/2] Total cells across all partitions: {total_cells:,}")

    # Adjust n_sample if total is smaller
    if total_cells < n_sample:
        warnings.warn(
            f"Requested {n_sample:,} cells, but only {total_cells:,} available. "
            f"Will sample all {total_cells:,} cells instead."
        )
        n_sample = total_cells

    # 2) Choose n_sample random "global" indices
    sampled_global_indices = np.sort(np.random.choice(total_cells, size=n_sample, replace=False))
    print(f"[1/2] Chosen {len(sampled_global_indices):,} random indices.")

    # 3) Second pass: read only the needed rows from each partition
    print(f"[2/2] Loading partitions again and extracting sampled rows...")
    sampled_adatas = []
    start_idx = 0
    idx_ptr = 0  # pointer into sampled_global_indices
    n_global = len(sampled_global_indices)

    for i, fname in enumerate(partition_files):
        path = os.path.join(partitions_dir, fname)
        count_i = partition_cell_counts[i]
        end_idx = start_idx + count_i  # partition i covers [start_idx..end_idx)

        # Figure out which random indices fall in [start_idx, end_idx)
        local_indices = []
        while idx_ptr < n_global:
            gidx = sampled_global_indices[idx_ptr]
            if gidx < start_idx:
                idx_ptr += 1
                continue
            if gidx >= end_idx:
                break
            # It's in range
            local_indices.append(gidx - start_idx)
            idx_ptr += 1

        if local_indices:
            # Now we load the entire partition in memory
            ad_full = sc.read_h5ad(path)  # no 'backed' => fully in memory
            sub = ad_full[local_indices, :].copy()
            sampled_adatas.append(sub)

            # Clean up
            del sub
            del ad_full

        start_idx = end_idx
        if idx_ptr >= n_global:
            break

    print(f"[2/2] Concatenating partial samples from {len(sampled_adatas)} partitions...")
    if len(sampled_adatas) == 1:
        final_adata = sampled_adatas[0]
    else:
        # Use anndata's built-in concat
        final_adata = sampled_adatas[0].concatenate(
            *sampled_adatas[1:], join='outer', batch_key=None
        )

    print(f"[2/2] Final sample shape: {final_adata.n_obs} cells x {final_adata.n_vars} vars")
    final_adata.write_h5ad(output_file)
    print(f"[DONE] Wrote random sample of {final_adata.n_obs:,} cells to {output_file}")


if __name__ == "__main__":
    main()