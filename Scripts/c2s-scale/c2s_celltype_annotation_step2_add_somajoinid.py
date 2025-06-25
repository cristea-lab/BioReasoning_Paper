#!/usr/bin/env python3
"""
Merge soma_joinid into C2S predictions NDJSON.

Usage:
    python c2s_celltype_annotation_step2_add_somajoinid.py \
        --input_h5ad /path/to/raw_data.h5ad \
        --input_ndjson /path/to/c2s_scale_results.ndjson \
        --output_ndjson /path/to/c2s_scale_results_with_soma.ndjson
"""

import argparse
import json
import anndata

def main():
    parser = argparse.ArgumentParser(
        description="Merge soma_joinid into C2S NDJSON predictions"
    )
    parser.add_argument(
        "--input_h5ad",
        type=str,
        required=True,
        help="Original input h5ad file containing soma_joinid"
    )
    parser.add_argument(
        "--input_ndjson",
        type=str,
        required=True,
        help="C2S predictions ndjson file"
    )
    parser.add_argument(
        "--output_ndjson",
        type=str,
        required=True,
        help="Output ndjson with soma_joinid field added"
    )
    args = parser.parse_args()

    print(f"Loading AnnData from {args.input_h5ad}")
    adata = anndata.read_h5ad(args.input_h5ad)
    if "soma_joinid" not in adata.obs.columns:
        raise KeyError("'soma_joinid' column not found in adata.obs")
    soma_ids = adata.obs["soma_joinid"].astype(str).tolist()
    print(f"Found {len(soma_ids)} soma_joinid entries")

    print(f"Reading predictions from {args.input_ndjson}")
    preds = []
    with open(args.input_ndjson, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            preds.append(json.loads(line))
    print(f"Read {len(preds)} prediction records")

    # Merge by index
    count = min(len(soma_ids), len(preds))
    if len(soma_ids) != len(preds):
        print(f"Warning: number of predictions ({len(preds)}) != number of soma IDs ({len(soma_ids)}); merging first {count} entries")

    print(f"Merging {count} records...")
    with open(args.output_ndjson, 'w', encoding='utf-8') as fout:
        for i in range(count):
            rec = preds[i]
            rec["soma_joinid"] = soma_ids[i]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved merged NDJSON with soma_joinid to {args.output_ndjson}")

if __name__ == '__main__':
    main()