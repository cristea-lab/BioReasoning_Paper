#!/usr/bin/env bash
#
# Usage:
#   ./run_pipeline.sh INPUT_NDJSON OUTPUT_DIR
#
# Example:
#   ./run_pipeline.sh my_cells.ndjson output_folder
#   # Intermediate files: output_folder/step1_extracted.ndjson, output_folder/step2_matched.ndjson
#   # Final outputs:      output_folder/results.csv,           output_folder/results.json
#

set -euo pipefail

# Check for required arguments
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_NDJSON OUTPUT_DIR"
  exit 1
fi

INPUT_NDJSON="$1"
OUTPUT_DIR="$2"

# Create output directory if it doesn’t exist
mkdir -p "${OUTPUT_DIR}"

# Intermediate file names
STEP1_OUT="${OUTPUT_DIR}/step1_extracted.ndjson"
STEP2_OUT="${OUTPUT_DIR}/step2_matched.ndjson"

# Final output filenames
CSV_OUT="${OUTPUT_DIR}/results.csv"
JSON_OUT="${OUTPUT_DIR}/results.json"

# Path to your python scripts in the subdirectory
SCRIPT_DIR="python_scripts"

echo "==== 1) Extracting predicted cell type from 'response' field ===="
python3 "${SCRIPT_DIR}/a_extract_celltype.py" \
  "${INPUT_NDJSON}" \
  "${STEP1_OUT}"

echo "==== 2) Matching labels to CL ontology via EBI OLS ===="
python3 "${SCRIPT_DIR}/b_match_ols_ontology.py" \
  --input_ndjson "${STEP1_OUT}" \
  --output_ndjson "${STEP2_OUT}" \
  --parallel \
  --max_workers 50

echo "==== 3) Evaluating predictions via multi-level descendants in Ubergraph ===="
python3 "${SCRIPT_DIR}/c_ubergraph_eval_all_descendants.py" \
  --input_ndjson "${STEP2_OUT}" \
  --output_csv  "${CSV_OUT}" \
  --output_json "${JSON_OUT}"

echo "==== Pipeline Complete ===="
echo "Outputs:"
echo "  - Intermediate 1: ${STEP1_OUT}"
echo "  - Intermediate 2: ${STEP2_OUT}"
echo "  - Final CSV:      ${CSV_OUT}"
echo "  - Final JSON:     ${JSON_OUT}"