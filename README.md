# Biological Reasoning with Reinforcement Learning through Natural Language Enables Generalizable Zero‑Shot Cell Type Annotations

Code to fully reproduce the experiments from **Wang *****et al.***** (2025)** demonstrating that the 671 B‑parameter **DeepSeek‑R1-0528** language model performs accurate and generalizable zero‑shot cell‑type annotation from scRNA‑seq data while providing interpretable biological reasoning.

## Repository layout

```
Scripts/
├── api_calls/               # Helpers to query DeepSeek‑R1, GPT‑4o and DeepSeek‑V3
├── data_generation/         # Exactly how we download, sample, and format given public data
├── prompt_generations/      # Generate ranked marker‑gene prompts from .h5ad or clustering results
├── scTab/                   # Baseline: scTab inference
├── scGPT/                   # Baseline: scGPT inference
├── evaluation/              # Harmonisation + ontology‑aware evaluation pipeline
└── plots/                   # Figure generation
```

*The **`.git`** directory is included only to mirror the published GitHub history.*

## Quick start

1. **Clone the repo**

```bash
git clone https://github.com/cristea-lab/BioReasoning_Paper
cd BioReasoning_Paper
```

2. **Create a conda env (Python ≥ 3.10) and install dependencies**

```bash
conda create -n bio_reasoning python=3.10
conda activate bio_reasoning
pip install numpy pandas scanpy anndata torch scgpt cellnet networkx rdflib tqdm openai
```

3. **Download pre‑computed outputs (optional but recommended)**\
   Prompts, model completions and evaluation tables can be retrieved from Google Drive (\~1 GB of ZIP file):

[https://drive.google.com/drive/folders/1egpLkd91Rlovhtrze2TW2-d29kz82_9J?usp=sharing](https://drive.google.com/drive/folders/1egpLkd91Rlovhtrze2TW2-d29kz82_9J?usp=sharing)

4. **Run a full pipeline on your own **``** file**

```bash
# 1. Generate prompt
python Scripts/prompt_generations/prompt_generation_from_h5ad.py \
       --input my_data.h5ad \
       --output prompts/

# 2. Query DeepSeek‑R1 (requires DEEPSEEK_API_KEY env variable)
python Scripts/api_calls/QueryDeepseek.py \
       --input prompts/cluster_markers.ndjson \
       --output completions/

# 3. Ontology‑aware evaluation (requires ground‑truth labels)
bash Scripts/evaluation/run_pipelines.sh completions/predictions.ndjson results/
```

Each script has `--help` for the full list of options.

Note: there are also modified versions available under the same folder. Ex. generating prompts from a list of given cell type labels (termed "classifier" mode in the manuscript).

## Reproducing the paper figures

After generating the evaluation tables:

```bash
jupyter lab
# open the notebooks in Scripts/plots/
```

## Minimal dependencies

- Python 3.10+
- `numpy`, `pandas`, `scanpy`, `anndata`
- `torch` (>= 2.2, CUDA optional but recommended)
- `scgpt`, `cellnet` (for baselines)
- `networkx`, `rdflib` (Ubergraph evaluation)
- `tqdm`, `openai`, `pyyaml`

## Citing

If you use this code, please cite:

```bibtex
@article{wang2025bioreasoning,
  title   = {Biological Reasoning with Reinforcement Learning through Natural Language Enables Generalizable Zero‑Shot Cell Type Annotations},
  author  = {Wang, Xi and Tan, Runzi and Wang, Bo and Cristea, Simona},
  year    = {2025},
  journal = {bioRxiv},
}
```

## License

MIT — see `LICENSE` for details.

