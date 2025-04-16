#!/usr/bin/env python3

import json
import sys

"""
Usage:
    python clean_output.py old_file.ndjson new_file.ndjson

This will read each line from old_file.ndjson,
rename "output_text" to "response" if present,
and write out new_file.ndjson.
"""

def rename_output_text_to_response(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # skip invalid lines
                continue

            # If "output_text" exists, rename it
            if "output_text" in data:
                data["response"] = data["output_text"]
                del data["output_text"]

            # Optionally: if "error" also needs to be changed or removed, do it here:
            # e.g.: if "error" in data: del data["error"]

            # Write out the updated line
            fout.write(json.dumps(data) + "\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: python clean_output.py old_file.ndjson new_file.ndjson")
        sys.exit(1)

    in_file = sys.argv[1]
    out_file = sys.argv[2]
    rename_output_text_to_response(in_file, out_file)
    print(f"Done. Wrote renamed lines to {out_file}")

if __name__ == "__main__":
    main()