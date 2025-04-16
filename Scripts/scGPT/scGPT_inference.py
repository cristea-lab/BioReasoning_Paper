'''
This script processes an H5AD file using the scGPT model to get cell embeddings and predict cell types using a FAISS index.
It filters genes based on a vocabulary, selects the top 3000 highly variable genes, gets cell embeddings, predicts cell types,
and saves the results along with the true cell types in an NDJSON file.

Usage:
    python script.py --input_h5ad path/to/input.h5ad --output_ndjson path/to/output.ndjson

Requirements:
    - The input H5AD file must have 'obs.cell_type' and 'var.feature_name'.
    - The scGPT model and FAISS index must be properly set up in the specified directories.
'''

### Imports
import argparse
from pathlib import Path
import json
import numpy as np
import scanpy as sc
from tqdm import tqdm

import scgpt as scg
from scgpt.tokenizer.gene_tokenizer import GeneVocab

import gpu_utils
gpu_utils.set_gpu()

from build_atlas_index_faiss import load_index, vote

### Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Process H5AD file with scGPT and save predictions to NDJSON.")
    parser.add_argument("--input_h5ad", required=True, help="Path to the input H5AD file.")
    parser.add_argument("--output_ndjson", required=True, help="Path to the output NDJSON file.")
    return parser.parse_args()

### Function Definitions
def filter_genes(adata, model_dir, gene_col):
    """Filter genes in adata to those present in the scGPT vocabulary."""
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)
    id_in_vocab = [1 if gene in vocab else -1 for gene in adata.var[gene_col]]
    gene_ids_in_vocab = np.array(id_in_vocab)
    print(
        f"Match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    return adata[:, gene_ids_in_vocab >= 0]

def predict_celltype(embed_adata, index, meta_labels, k=50):
    """Predict cell types using FAISS nearest neighbor search and majority voting."""
    distances, idx = index.search(embed_adata.obsm["X_scGPT"], k)
    predict_labels = meta_labels[idx]
    voting = []
    for preds in tqdm(predict_labels, desc="Voting on predictions"):
        voting.append(vote(preds, return_prob=False)[0])
    return voting

def save_scGPT_results(ndjson_path, adata, preds):
    """Save predictions and true cell types to an NDJSON file."""
    preds_json = []
    for i in range(len(preds)):
        preds_json.append({
            'soma_joinid': str(adata.obs['soma_joinid'][i]),
            'cell_type_ground_truth': adata.obs['cell_type'][i],
            'response': f"Cell type: {preds[i]}"
        })
    
    with open(ndjson_path, "w") as outfile:
        for obj in preds_json:
            outfile.write(json.dumps(obj) + "\n")

### Main Execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    input_h5ad = args.input_h5ad
    output_ndjson = args.output_ndjson

    # Define static paths and variables
    model_dir = Path('/cristealab/rtan/scGPT/models/whole_human')
    index_dir = Path("/cristealab/rtan/scGPT/DSR1/data/CellXGene_faiss_index/")
    gene_col = 'feature_name'

    # Load FAISS index
    index, meta_labels = load_index(
        index_dir=index_dir,
        use_config_file=False,
        use_gpu=True,
    )
    print(f"Loaded index with {index.ntotal} cells")

    # Load H5AD file
    adata = sc.read_h5ad(input_h5ad)
    print(f"Loaded data with {adata.shape[0]} cells and {adata.shape[1]} genes.")

    # Validate required columns
    if 'cell_type' not in adata.obs.columns:
        raise ValueError("The input H5AD file must have 'obs.cell_type'.")
    if gene_col not in adata.var.columns:
        raise ValueError(f"The input H5AD file must have 'var.{gene_col}'.")

    # Process data
    # Filter genes based on vocabulary
    adata = filter_genes(adata, model_dir, gene_col)
    print(f"After filtering, {adata.shape[1]} genes remain.")

    # Select top 3000 highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, flavor='seurat_v3')
    adata = adata[:, adata.var.highly_variable]
    print("Selected 3000 highly variable genes.")

    # Get cell embeddings using scGPT
    embed_adata = scg.tasks.embed_data(
        adata,
        model_dir,
        gene_col=gene_col,
        cell_type_key='cell_type',
        batch_size=64,
    )
    print("Computed cell embeddings.")

    # Predict cell types
    preds = predict_celltype(embed_adata, index, meta_labels, k=50)
    print("Predicted cell types.")

    # Save results to NDJSON
    save_scGPT_results(output_ndjson, embed_adata, preds)
    print(f"Saved results to {output_ndjson}.")