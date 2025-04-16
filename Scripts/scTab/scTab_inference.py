#!/usr/bin/env python3
#!/usr/bin/env python3

'''
Script: scTab Cell Type Prediction

Description:
This script runs the scTab deep learning model to predict cell types from single-cell RNA-seq data.
It takes an AnnData object (.h5ad file) with raw counts, processes the data, runs inference,
and saves predictions in .ndjson format.

Prerequisites:
- Python 3.10+
- Required packages: anndata, numpy, pandas, torch, scanpy, scipy, tqdm, pyyaml
- cellnet package installed
- scTab model files in '/cristealab/rtan/scGPT/DSR1/sctab/' directory relative to script:
  - '/cristealab/rtan/scGPT/DSR1/sctab/merlin_cxg_2023_05_15_sf-log1p_minimal/var.parquet'
  - '/cristealab/rtan/scGPT/DSR1/sctab/scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt'
  - '/cristealab/rtan/scGPT/DSR1/sctab/scTab-checkpoints/scTab/run5/hparams.yaml'
  - '/cristealab/rtan/scGPT/DSR1/sctab/merlin_cxg_2023_05_15_sf-log1p_minimal/categorical_lookup/cell_type.parquet'

Usage:
python script.py --input_h5ad INPUT_FILE --output_ndjson OUTPUT_FILE [OPTIONS]

Arguments:
  --input_h5ad        Path to input .h5ad file with raw counts (required)
  --output_ndjson     Path to save predictions in .ndjson format (required)
  --gpu_core         GPU core to use (default: '0'), use 'cpu' to force CPU
  --feature_name     Column in adata.var with gene symbols (default: 'feature_name'),
                     use 'index' if gene names are in index

Examples:
  # Basic usage with default GPU
  python script.py --input_h5ad data.h5ad --output_ndjson predictions.ndjson

  # Use specific GPU core
  python script.py --input_h5ad data.h5ad --output_ndjson predictions.ndjson --gpu_core 1

  # Force CPU usage
  python script.py --input_h5ad data.h5ad --output_ndjson predictions.ndjson --gpu_core cpu

  # Custom gene name column
  python script.py --input_h5ad data.h5ad --output_ndjson predictions.ndjson --feature_name gene_id

Input Requirements:
- .h5ad file with raw counts in adata.X
- Gene symbols in adata.var (specified column or index)
- 'soma_joinid' and 'cell_type' columns in adata.obs

Output Format:
- .ndjson file where each line is a JSON object with:
  - soma_joinid: Cell identifier
  - cell_type_ground_truth: Original annotation
  - response: Predicted cell type ("Cell type: [prediction]")

Process:
1. Loads input data and model gene references
2. Filters and aligns genes
3. Normalizes data (10,000 counts per cell + log1p)
4. Runs scTab model inference
5. Maps predictions to cell type labels
6. Saves results

Notes:
- Progress is shown with a tqdm progress bar
- Falls back to CPU if GPU is unavailable
- Requires sufficient memory for the input data size
'''
import argparse
import json
import os
from collections import OrderedDict

import anndata
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from cellnet.utils.data_loading import streamline_count_matrix, dataloader_factory
from cellnet.tabnet.tab_network import TabNet

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run scTab model inference on single-cell RNA-seq data to predict cell types.'
    )
    parser.add_argument(
        '--input_h5ad',
        type=str,
        required=True,
        help='Path to input .h5ad file containing raw counts in adata.X'
    )
    parser.add_argument(
        '--output_ndjson',
        type=str,
        required=True,
        help='Path to output .ndjson file for saving predictions'
    )
    parser.add_argument(
        '--gpu_core',
        type=str,
        default='0',
        help='GPU core to use (default: 0). Set to "cpu" to force CPU usage'
    )
    parser.add_argument(
        '--feature_name',
        type=str,
        default='feature_name',
        help='Column name in adata.var containing gene symbols (default: "feature_name"). Use "index" if gene names are in index'
    )
    return parser.parse_args()

def sf_log1p_norm(x):
    """Normalize each cell to have 10000 counts and apply log(x+1) transform."""
    counts = torch.sum(x, dim=1, keepdim=True)
    counts += counts == 0.  # avoid zero division
    scaling_factor = 10000. / counts
    return torch.log1p(scaling_factor * x)

def load_ndjson(file):
    """Load ndjson file into a list of dictionaries."""
    output = []
    with open(file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    single_cell = json.loads(line)
                    output.append(single_cell)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from line: {e}")
    return output

def save_sctab_results(ndjson_path, adata, preds):
    """
    Save scTab predictions in .ndjson format.
    
    Args:
        ndjson_path (str): Path to save output .ndjson file
        adata (anndata.AnnData): Annotated data object with observations
        preds (np.ndarray): Array of predicted cell type labels
    """
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

def main():
    """Main function to run scTab inference."""
    args = parse_args()

    # Set device
    if args.gpu_core != 'cpu' and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_core
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if args.gpu_core != 'cpu':
            print('CUDA not available, falling back to CPU')

    # Load data
    # All input data has raw counts in adata.X
    adata = anndata.read_h5ad(args.input_h5ad)
    print(f"Loaded data with shape: {adata.shape}")

    # Load model gene order
    genes_from_model = pd.read_parquet('/cristealab/rtan/scGPT/DSR1/sctab/merlin_cxg_2023_05_15_sf-log1p_minimal/var.parquet')

    # Streamline feature space
    gene_col = adata.var.index if args.feature_name == 'index' else adata.var[args.feature_name]
    adata = adata[:, gene_col.isin(genes_from_model.feature_name).to_numpy()]
    gene_col = adata.var.index if args.feature_name == 'index' else adata.var[args.feature_name]
    print(f"After gene filtering: {adata.shape}")

    x_streamlined = streamline_count_matrix(
        csc_matrix(adata.X),
        gene_col,
        genes_from_model.feature_name
    )
    print(f"Streamlined matrix shape: {x_streamlined.shape}")

    # Create data loader
    loader = dataloader_factory(x_streamlined, batch_size=2048)

    # Load checkpoint
    ckpt_path = '/cristealab/rtan/scGPT/DSR1/sctab/scTab-checkpoints/scTab/run5/val_f1_macro_epoch=41_val_f1_macro=0.847.ckpt'
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Extract tabnet weights
    tabnet_weights = OrderedDict()
    for name, weight in ckpt['state_dict'].items():
        if 'classifier.' in name:
            tabnet_weights[name.replace('classifier.', '')] = weight

    # Load model parameters
    with open('/cristealab/rtan/scGPT/DSR1/sctab/scTab-checkpoints/scTab/run5/hparams.yaml') as f:
        model_params = yaml.full_load(f.read())

    # Initialize model
    tabnet = TabNet(
        input_dim=model_params['gene_dim'],
        output_dim=model_params['type_dim'],
        n_d=model_params['n_d'],
        n_a=model_params['n_a'],
        n_steps=model_params['n_steps'],
        gamma=model_params['gamma'],
        n_independent=model_params['n_independent'],
        n_shared=model_params['n_shared'],
        epsilon=model_params['epsilon'],
        virtual_batch_size=model_params['virtual_batch_size'],
        momentum=model_params['momentum'],
        mask_type=model_params['mask_type'],
    )
    tabnet.load_state_dict(tabnet_weights)
    tabnet.eval()
    tabnet.to(device)

    # Run inference
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            x_input = sf_log1p_norm(batch[0]['X']).to(device)
            logits, _ = tabnet(x_input)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    preds = np.hstack(preds)

    # Map predictions to cell types
    cell_type_mapping = pd.read_parquet('/cristealab/rtan/scGPT/DSR1/sctab/merlin_cxg_2023_05_15_sf-log1p_minimal/categorical_lookup/cell_type.parquet')
    preds = cell_type_mapping.loc[preds]['label'].to_numpy()
    print(f"Generated {len(preds)} predictions")

    # Save results
    save_sctab_results(args.output_ndjson, adata, preds)
    print(f"Results saved to {args.output_ndjson}")

if __name__ == "__main__":
    main()