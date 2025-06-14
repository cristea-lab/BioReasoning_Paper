#!/usr/bin/env bash
#
# Usage:
#   ./run_pipeline.sh INPUT_NDJSON OUTPUT_DIR
#
# Example:
#   ./run_pipeline.sh my_cells.ndjson output_folder
#   # Intermediate files: output_folder/step1_extracted.ndjson, output_folder/step2_matched.ndjson
#   # Final outputs:      output_folder/results.csv, results.json, results_cleaned_v2.csv
#

set -euo pipefail

# -------------- argument parsing ------------------------------------------------
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_NDJSON OUTPUT_DIR"
  exit 1
fi

INPUT_NDJSON="$1"
OUTPUT_DIR="$2"

# -------------- set up paths ----------------------------------------------------
mkdir -p "${OUTPUT_DIR}"

# Intermediates
STEP1_OUT="${OUTPUT_DIR}/step1_extracted.ndjson"
STEP2_OUT="${OUTPUT_DIR}/step2_matched.ndjson"

# Final outputs from step 3
CSV_OUT="${OUTPUT_DIR}/results.csv"
JSON_OUT="${OUTPUT_DIR}/results.json"

# Output from new post-processing step 4
CLEAN_CSV_OUT="${OUTPUT_DIR}/results_cleaned_v2.csv"

# Where the Python scripts live
SCRIPT_DIR="python_scripts"

# -------------- 1) extract model prediction -------------------------------------
echo "==== 1) Extracting predicted cell type from 'response' field ===="
python3 "${SCRIPT_DIR}/a_extract_celltype.py" \
  "${INPUT_NDJSON}" \
  "${STEP1_OUT}"

# -------------- 2) match to Cell Ontology via EBI OLS ----------------------------
echo "==== 2) Matching labels to CL ontology via EBI OLS ===="
python3 "${SCRIPT_DIR}/b_match_ols_ontology.py" \
  --input_ndjson  "${STEP1_OUT}" \
  --output_ndjson "${STEP2_OUT}" \
  --parallel \
  --max_workers 50

# -------------- 3) ontology-aware evaluation (Ubergraph) -------------------------
echo "==== 3) Evaluating predictions via multi-level descendants in Ubergraph ===="
python3 "${SCRIPT_DIR}/c_ubergraph_eval_all_descendants.py" \
  --input_ndjson "${STEP2_OUT}" \
  --output_csv   "${CSV_OUT}" \
  --output_json  "${JSON_OUT}"

# -------------- 4) post-process CSV, add bootstrap + Wilson CIs ------------------
echo "==== 4) Post-processing results and adding confidence intervals ===="
python3 "${SCRIPT_DIR}/d_remove_empty_match_and_bootstrap.py" \
  --input_csv  "${CSV_OUT}" \
  --output_csv "${CLEAN_CSV_OUT}" \
  --n_boot 10000 \
  --alpha 0.05 \
  --seed 0

# -------------- done ------------------------------------------------------------
echo "==== Pipeline Complete ===="
echo "Outputs:"
echo "  - Intermediate 1: ${STEP1_OUT}"
echo "  - Intermediate 2: ${STEP2_OUT}"
echo "  - Evaluation CSV: ${CSV_OUT}"
echo "  - Evaluation JSON:${JSON_OUT}"
echo "  - Clean  CSV v2:  ${CLEAN_CSV_OUT}"