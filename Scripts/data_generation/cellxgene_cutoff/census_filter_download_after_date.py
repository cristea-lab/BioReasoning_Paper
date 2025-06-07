#!/usr/bin/env python3
"""
Example usage:

(1) Build index for 'normal' unique cells that are new compared to 2023-05-15:
    python census_filter_download.py build-index --filter-name normal --index-dir ./indexes

(2) Download those unique normal cells in chunks, with resume:
    python census_filter_download.py download-chunks \
        --filter-name normal \
        --index-dir ./indexes \
        --output-dir ./normal_unique_h5ad \
        --chunk-size 20000 \
        --resume

(3) For 'cancer' unique cells, provide a cancer_list.txt and do similarly:
    # Note: --cancer-list-file must come *before* the subcommand name!
    python census_filter_download.py --cancer-list-file cancer_list.txt build-index \
        --filter-name cancer \
        --index-dir ./indexes

    python census_filter_download.py --cancer-list-file cancer_list.txt download-chunks \
        --filter-name cancer \
        --index-dir ./indexes \
        --output-dir ./cancer_unique_h5ad \
        --resume
"""

import argparse
import os
import time
from tqdm import tqdm
import cellxgene_census

# Old release version (the one scGPT was trained on):
OLD_CENSUS_VERSION = "2023-05-15"

# New release version (the one you want to benchmark against):
NEW_CENSUS_VERSION = "2024-07-01"

def build_value_filter(filter_name, cancer_list_file=None):
    """
    Return a TileDB QueryCondition string for the given filter_name.
    We also require is_primary_data == True to ensure only unique cells.
    """
    if filter_name == "normal":
        # Normal, unique cells
        return "suspension_type != 'na' and is_primary_data == True and disease == 'normal'"

    elif filter_name == "cancer":
        if not cancer_list_file:
            raise ValueError("Must provide --cancer-list-file when filter-name='cancer'.")
        with open(cancer_list_file, "r") as f:
            diseases = [line.strip() for line in f if line.strip()]

        # Build the OR condition for cancer diseases
        or_clauses = [f"(disease == '{d}')" for d in diseases]
        disease_condition = " or ".join(or_clauses)

        # Add is_primary_data == True to exclude duplicates
        return f"suspension_type != 'na' and is_primary_data == True and ({disease_condition})"

    else:
        raise ValueError(f"Unknown filter name '{filter_name}'. Choose 'normal' or 'cancer'.")

def build_index(args):
    """
    Sub-command: build-index
    - Reads the 'old' release (2023-05-15) and 'new' release (2024-07-01),
      collecting soma_joinid for the same value_filter.
    - Writes only the set difference (new but not old) to the output .idx file.
    - Prints how many were in the old release, new release, and final difference.
    """
    filter_name = args.filter_name
    value_filter = build_value_filter(filter_name, cancer_list_file=args.cancer_list_file)

    index_file = os.path.join(args.index_dir, f"{filter_name}.idx")
    os.makedirs(args.index_dir, exist_ok=True)

    print(f"[build-index] Building index for filter '{filter_name}'")
    print(f"[build-index]   Old release: {OLD_CENSUS_VERSION}")
    print(f"[build-index]   New release: {NEW_CENSUS_VERSION}")
    print(f"[build-index]   Value filter: {value_filter}")
    print(f"[build-index]   Output .idx file -> {index_file}")

    # --- 1) Get old release soma_joinids ---
    with cellxgene_census.open_soma(census_version=OLD_CENSUS_VERSION) as old_census:
        old_obs = old_census["census_data"]["homo_sapiens"].obs.read(
            value_filter=value_filter,
            column_names=["soma_joinid"]
        ).concat()
    old_df = old_obs.to_pandas()
    old_ids = set(old_df["soma_joinid"].astype(int).tolist())
    print(f"[build-index] Found {len(old_ids):,} matching cells in old release ({OLD_CENSUS_VERSION}).")

    # --- 2) Get new release soma_joinids ---
    with cellxgene_census.open_soma(census_version=NEW_CENSUS_VERSION) as new_census:
        new_obs = new_census["census_data"]["homo_sapiens"].obs.read(
            value_filter=value_filter,
            column_names=["soma_joinid"]
        ).concat()
    new_df = new_obs.to_pandas()
    new_ids = set(new_df["soma_joinid"].astype(int).tolist())
    print(f"[build-index] Found {len(new_ids):,} matching cells in new release ({NEW_CENSUS_VERSION}).")

    # --- 3) Subtract old IDs from new ---
    diff_ids = new_ids - old_ids
    print(f"[build-index] 'New only' cell_ids = {len(diff_ids):,} (in new but NOT in old).")

    # --- 4) Write them to .idx file ---
    with open(index_file, "w") as f:
        for cid in diff_ids:
            f.write(f"{cid}\n")

    print(f"[build-index] Wrote index file with {len(diff_ids):,} lines: {index_file}")


def download_chunks(args):
    """
    Sub-command: download-chunks
    - Reads .idx file, partitions the IDs, downloads each partition as .h5ad,
      logs each cell ID -> partition file in a CSV.
    - If --resume is set, skip any partition that already has a .h5ad file.
    - Prints how many total lines were written to the log in the end.
    """
    filter_name = args.filter_name
    index_file = os.path.join(args.index_dir, f"{filter_name}.idx")
    output_dir = args.output_dir
    chunk_size = args.chunk_size
    resume = args.resume

    if not os.path.isfile(index_file):
        raise FileNotFoundError(f"Index file not found: {index_file}")

    # For the "cancer" filter, check if we have a cancer_list_file
    if filter_name == "cancer" and not args.cancer_list_file:
        raise ValueError("--cancer-list-file is required when using filter-name='cancer'.")

    os.makedirs(output_dir, exist_ok=True)

    print(f"[download-chunks] Reading index file: {index_file}")
    with open(index_file, "r") as f:
        cell_id_strs = f.read().strip().split()
    cell_ids = list(map(int, cell_id_strs))

    total_cells = len(cell_ids)
    print(f"[download-chunks] Total {total_cells:,} cells to download for filter '{filter_name}'.")

    partition_count = (total_cells + chunk_size - 1) // chunk_size
    print(f"[download-chunks] Partitioning into {partition_count} chunk(s), up to {chunk_size} cells each.")

    master_log_path = os.path.join(output_dir, f"{filter_name}_cell2file_log.csv")
    print(f"[download-chunks] Master log: {master_log_path}")

    # We'll keep track of how many lines we write to this log
    written_count = 0

    log_mode = "a" if (resume and os.path.exists(master_log_path)) else "w"
    with open(master_log_path, log_mode) as logf:
        if log_mode == "w":
            logf.write("soma_joinid,partition_file\n")

        # Open the NEW census (since that's the data we are retrieving)
        with cellxgene_census.open_soma(census_version=NEW_CENSUS_VERSION) as census, tqdm(
            total=partition_count, desc="Downloading partitions", unit="part"
        ) as pbar:

            for i in range(partition_count):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_cells)
                part_cell_ids = cell_ids[start_idx:end_idx]

                partition_filename = f"partition_{i}.h5ad"
                partition_path = os.path.join(output_dir, partition_filename)

                # If resuming and file exists, skip
                if resume and os.path.exists(partition_path):
                    pbar.update(1)
                    print(f"Skipping partition {i+1}/{partition_count} because {partition_filename} already exists.")
                    continue

                t0 = time.time()
                adata = cellxgene_census.get_anndata(
                    census=census,
                    organism="Homo sapiens",
                    obs_coords=part_cell_ids
                )
                t1 = time.time()

                # Write .h5ad
                adata.write_h5ad(partition_path)
                t2 = time.time()

                # Log the cell IDs -> partition file
                for cid in part_cell_ids:
                    logf.write(f"{cid},{partition_filename}\n")
                    written_count += 1

                # Approx speed
                file_size_bytes = os.path.getsize(partition_path)
                download_duration = t1 - t0
                write_duration = t2 - t1
                total_duration = t2 - t0
                file_size_mb = file_size_bytes / (1024**2)
                approx_speed = file_size_mb / download_duration if download_duration > 0 else 0

                print(f"\nPartition {i+1}/{partition_count} -> {partition_path}")
                print(f"  Cells: {len(part_cell_ids):,}")
                print(f"  .h5ad size: {file_size_mb:.2f} MB")
                print(f"  Time (download): {download_duration:.2f}s")
                print(f"  Time (write): {write_duration:.2f}s")
                print(f"  Time (total): {total_duration:.2f}s")
                print(f"  Approx download speed: {approx_speed:.2f} MB/s")

                pbar.update(1)

    print("[download-chunks] Finished partitions.")
    print(f"[download-chunks] Wrote {written_count:,} lines total to log: {master_log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download only the 'new' unique cells from 2024-07-01 Census that were NOT in 2023-05-15."
    )
    parser.add_argument("--cancer-list-file", type=str, default=None,
                        help="File with disease terms if filter-name='cancer'.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # build-index
    parser_index = subparsers.add_parser("build-index",
        help="Build .idx file containing 'new only' soma_joinids."
    )
    parser_index.add_argument("--filter-name", required=True, choices=["normal","cancer"],
                              help="Choose 'normal' or 'cancer' (unique cells).")
    parser_index.add_argument("--index-dir", required=True, help="Directory for the .idx file.")
    parser_index.set_defaults(func=build_index)

    # download-chunks
    parser_dl = subparsers.add_parser("download-chunks",
        help="Download partitioned .h5ad files for the .idx list, with optional --resume."
    )
    parser_dl.add_argument("--filter-name", required=True, choices=["normal","cancer"],
                           help="Same name used in build-index.")
    parser_dl.add_argument("--index-dir", required=True, help="Directory where the .idx file is located.")
    parser_dl.add_argument("--output-dir", required=True, help="Where to store .h5ad partitions.")
    parser_dl.add_argument("--chunk-size", type=int, default=20000,
                           help="Number of cells per partition. Default=20,000.")
    parser_dl.add_argument("--resume", action="store_true",
                           help="Skip partitions that already have a .h5ad file.")
    parser_dl.set_defaults(func=download_chunks)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()