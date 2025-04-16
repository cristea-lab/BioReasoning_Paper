#!/usr/bin/env python3
#
# Script Name: extract_cell_type.py
#
# Description:
# This script reads a file containing one JSON object per line (NDJSON format).
# Each JSON object is expected to have the following fields:
#   - "soma_joinid"
#   - "cell_type_ground_truth"
#   - "response"
#
# Within the "response" field, there is a line at the end formatted as:
#   "Cell type: X"
#
# The script will:
#  1) Parse each JSON object.
#  2) Extract the "soma_joinid" value.
#  3) Extract the "cell_type_ground_truth" value.
#  4) Extract the final cell type string (X) from the "response" field.
#  5) Write a new NDJSON (line-delimited JSON) file, where each line contains
#     an object of the form:
#       {
#         "soma_joinid": <value>,
#         "ground_truth": <value>,
#         "predicted_cell_type": <value>
#       }
#
# Usage:
#   ./extract_cell_type.py input.json output.json
#   (or "python3 extract_cell_type.py input.json output.json")
#
# Example:
#   Input line (NDJSON):
#   {
#     "soma_joinid": 14055810,
#     "cell_type_ground_truth": "cerebellar granule cell",
#     "response": "...Cell type: Cerebellar granule neuron"
#   }
#
#   Output line (NDJSON):
#   {
#     "soma_joinid": 14055810,
#     "ground_truth": "cerebellar granule cell",
#     "predicted_cell_type": "Cerebellar granule neuron"
#   }
#

import sys
import json
import re

def main():
    if len(sys.argv) != 3:
        print("Usage: {} <input_ndjson> <output_ndjson>".format(sys.argv[0]))
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue  # skip any empty lines

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # If there's a parsing error, skip this line or handle as needed
                continue

            # Extract required fields
            soma_joinid = data.get("soma_joinid", None)
            ground_truth = data.get("cell_type_ground_truth", None)
            response_text = data.get("response", "")

            # Use regex to capture what follows "Cell type:"
            match = re.search(r'Cell type:\s*(.+)', response_text)
            if match:
                predicted_cell_type = match.group(1).strip()
            else:
                predicted_cell_type = ""

            # Prepare output JSON object
            output_obj = {
                "soma_joinid": soma_joinid,
                "ground_truth": ground_truth,
                "predicted_cell_type": predicted_cell_type
            }

            # Write as NDJSON
            fout.write(json.dumps(output_obj, ensure_ascii=False))
            fout.write("\n")

if __name__ == "__main__":
    main()