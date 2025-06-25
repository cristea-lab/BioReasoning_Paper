#!/usr/bin/env python3
"""
Cell type annotation script using Cell2Sentence (C2S)

This script performs two main steps:
1. Preprocessing of a raw AnnData (.h5ad) file
2. Zero-shot cell type annotation inference using a C2S model

Usage:
    python c2s_celltype_annotation.py \
        --input_h5ad /path/to/raw_data.h5ad \
        --n_genes 500 \
        --cuda_device 0 \
        --output_dir /path/to/output_folder
"""

import os
import random
import numpy as np
from pathlib import Path
import json
import anndata
import scanpy as sc
import cell2sentence as cs
from cell2sentence.tasks import predict_cell_types_of_data
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Cell type annotation using C2S: preprocess and inference"
    )
    parser.add_argument(
        "--input_h5ad",
        type=str,
        required=True,
        help="Path to input h5ad file before preprocessing"
    )
    parser.add_argument(
        "--n_genes",
        type=int,
        default=500,
        help="Number of genes to use for inference"
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device ID to use (e.g., 0 for cuda:0)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store all results"
    )
    args = parser.parse_args()

    # Set CUDA device visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
    print(f"Using CUDA device: cuda:{args.cuda_device}")

    # Set random seed
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    print(f"Random seed set to {SEED}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory created/verified at {output_dir}")

    # ---------------- Preprocessing ----------------
    print("Starting preprocessing...")
    adata = anndata.read_h5ad(args.input_h5ad)
    print(f"Loaded raw data from {args.input_h5ad} with shape {adata.shape}")

    # Set variable names from feature_name column if available
    if "feature_name" in adata.var.columns:
        adata.var_names = adata.var["feature_name"].tolist()
        print("Set AnnData var_names from feature_name column")
    else:
        print("Warning: 'feature_name' column not found in var; using existing var_names.")

    # Annotate mitochondrial genes and calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True
    )
    print("Calculated QC metrics (including mitochondrial percentage)")

    # Normalize and log-transform
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata, base=10)
    print("Data normalized and log-transformed")

    # Keep only relevant metadata columns
    desired_pre_cols = ["cell_type", "tissue", "sex"]
    present_pre_cols = [col for col in desired_pre_cols if col in adata.obs.columns]
    missing_pre_cols = [col for col in desired_pre_cols if col not in adata.obs.columns]
    if missing_pre_cols:
        print(f"Warning: metadata columns not found and will be skipped: {missing_pre_cols}")
    adata.obs = adata.obs[present_pre_cols]
    print(f"Filtered metadata to keep columns: {present_pre_cols}")

    # Dimensionality reduction
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_pcs=50)
    sc.tl.umap(adata)
    print("Computed PCA, neighbors graph, and UMAP embedding")

    # Plot UMAPs
    try:
        sc.pl.umap(adata, color="tissue", size=8, title="UMAP: tissue")
        sc.pl.umap(adata, color="cell_type", size=8, title="UMAP: cell_type")
        print("UMAP plots (tissue and cell_type) generated")
    except Exception as e:
        print(f"Warning: UMAP plotting failed: {e}")

    # Save processed AnnData
    processed_path = output_dir / "processed_data.h5ad"
    adata.write_h5ad(processed_path)
    print(f"Saved processed AnnData to {processed_path}")

    # ---------------- Inference ----------------
    print("Starting inference...")
    adata = anndata.read_h5ad(processed_path)
    print(f"Re-loaded processed data from {processed_path}")

    # Add organism metadata
    adata.obs["organism"] = "Homo sapiens"

    # Prepare obs columns for inference
    desired_inf_cols = ["cell_type", "tissue", "organism", "sex"]
    present_inf_cols = [col for col in desired_inf_cols if col in adata.obs.columns]
    missing_inf_cols = [col for col in desired_inf_cols if col not in adata.obs.columns]
    if missing_inf_cols:
        print(f"Warning: metadata columns not found and will be skipped in inference: {missing_inf_cols}")
    adata.obs = adata.obs[present_inf_cols]
    print(f"Prepared obs for inference with columns: {present_inf_cols}")

    # Plot UMAP colored by cell_type if available
    try:
        sc.pl.umap(adata, color="cell_type", size=8, title="UMAP: cell_type (inference)")
        print("UMAP plot colored by cell_type generated for inference review")
    except Exception as e:
        print(f"Warning: inference UMAP plotting failed: {e}")

    # Prepare C2S data
    arrow_dir = output_dir / "c2s_data"
    arrow_dir.mkdir(parents=True, exist_ok=True)
    arrow_ds, vocab = cs.CSData.adata_to_arrow(
        adata=adata,
        random_state=SEED,
        sentence_delimiter=' ',
        label_col_names=present_inf_cols
    )
    print(f"Converted AnnData to Arrow dataset; vocabulary size: {len(vocab)}")

    csdata = cs.CSData.csdata_from_arrow(
        arrow_dataset=arrow_ds,
        vocabulary=vocab,
        save_dir=str(arrow_dir),
        save_name="c2s_scale",
        dataset_backend="arrow"
    )
    print(f"CSData saved to {arrow_dir}/c2s_scale.arrow files")

    # Initialize and run model
    model_path = "vandijklab/C2S-Scale-Pythia-1b-pt"
    infer_dir = output_dir / "c2s_scale_inference"
    infer_dir.mkdir(parents=True, exist_ok=True)

    csmodel = cs.CSModel(
        model_name_or_path=model_path,
        save_dir=str(infer_dir),
        save_name="cell_type_pred_pythia_1B_inference"
    )
    print(f"Loaded C2S model from {model_path}")

    preds = predict_cell_types_of_data(
        csdata=csdata,
        csmodel=csmodel,
        n_genes=args.n_genes
    )
    print(f"Completed inference on {len(preds)} cells using top {args.n_genes} genes")

    # Display sample predictions
    for pred, gt in zip(preds[::5], arrow_ds["cell_type"][::5]):
        if isinstance(pred, str) and pred.endswith('.'):
            pred = pred[:-1]
        print(f"Predicted: {pred} | Ground truth: {gt}")
    print("Sample predictions printed")

    # Save predictions
    result_dir = output_dir / "prediction_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    out_file = result_dir / "c2s_scale_results.ndjson"
    with out_file.open("w", encoding="utf-8") as f:
        for pred, gt in zip(preds, arrow_ds["cell_type"]):
            if isinstance(pred, str) and pred.endswith('.'):
                pred = pred[:-1]
            rec = {"cell_type_ground_truth": str(gt), "response": f"Cell type: {pred}"}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {len(preds)} prediction records to {out_file}")

if __name__ == "__main__":
    main()