#!/usr/bin/env python3
"""
Example Usage:

    python compare_cell_types_ndjson.py \
        --input_ndjson my_cells_normalized.ndjson \
        --output_csv comparison_results.csv \
        --output_json comparison_results.json

Description:
    1) Reads an NDJSON file where each line has at least:
         soma_joinid,
         ground_truth_label_clean,
         predicted_label_clean
       (These come from the first-pass script `ols_label_normalize_ndjson.py`.)
    2) Gathers unique ground-truth and predicted labels (the "clean" official labels).
    3) Queries Ubergraph to find **all descendants** (multi-level) of each label via `rdfs:subClassOf+`.
    4) For each line (cell), checks if:
         - predicted == ground_truth            => "SAME_LABEL"
         - predicted is any descendant of ground_truth => "PREDICTED_IS_CHILD_OF_GT"
         - ground_truth is any descendant of predicted => "GT_IS_CHILD_OF_PREDICTED"
         - otherwise => "NO_DIRECT_CHILD_RELATION"
       We then decide if scTab would call it "TRUE" or "FALSE".
    5) Outputs:
       - CSV with row-by-row + final metrics (if --output_csv).
       - JSON with row-by-row data + metrics (if --output_json).

WARNING: Using `rdfs:subClassOf+` can return large results if a label has
         many descendants. Performance may degrade for broad classes.
"""

import argparse
import json
import csv
from typing import List, Dict
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON

###################################
#   HELPER FUNCTIONS
###################################

def chunks(lst, n):
    """Batch a list into successive n-sized chunks."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def escape_sparql_string(s: str) -> str:
    """
    Escape backslashes and double-quotes to keep the label valid as a SPARQL string literal.
    """
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    return s

def get_all_descendants_query(value_term: str, label_list: List[str]) -> str:
    """
    Build a SPARQL query that retrieves ALL descendants of each label (multi-level),
    using rdfs:subClassOf+ ?parent.

    We do string-based matching for the parent label. If we pass '?cell_type', we do:
        VALUES ?cell_type { "rod bipolar cell" "astrocyte" ... }
    and then match rdfs:label to that text.

    For each matching parent class, we find all ?child such that:
        ?child rdfs:subClassOf+ ?parent
    """
    # Use double quotes around each label, properly escaped
    updated_list = [
        f"\"{escape_sparql_string(label)}\""
        for label in label_list
    ]

    # Build the query
    query = (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
        "PREFIX CL: <http://purl.obolibrary.org/obo/CL_>"
        "PREFIX owl: <http://www.w3.org/2002/07/owl#>"
        "SELECT * WHERE { "
        "  ?parent rdfs:label ?cell. "
        "  ?child rdfs:subClassOf+ ?parent. "
        "  ?child rdfs:label ?child_label. "
        "  ?parent rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>. "
        "  ?child rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>. "
        "  <http://purl.obolibrary.org/obo/cl/cl-base.owl> owl:versionIRI ?version. "
        f"  BIND(str(?cell) AS ?cell_type) VALUES {value_term} {{ {' '.join(updated_list)} }} "
        "}"
    )
    return query

def retrieve_all_descendants_from_ubergraph(label_list: List[str]) -> Dict[str, List[str]]:
    """
    For each label in 'label_list', find *all* descendant labels (multi-level)
    using rdfs:subClassOf+ queries.

    Returns: { parent_label : [descendant_label, ...], ... }

    If a label doesn't match exactly in CL, we get no descendants for that label.
    """
    desc_map = {}
    if not label_list:
        return desc_map

    sparql = SPARQLWrapper("https://ubergraph.apps.renci.org/sparql")
    sparql.method = 'POST'
    sparql.setReturnFormat(SPARQLJSON)

    value_term = "?cell_type"

    # We'll batch the labels to avoid huge queries in one go
    for batch in chunks(label_list, 80):
        query = get_all_descendants_query(value_term, batch)
        sparql.setQuery(query)
        ret = sparql.queryAndConvert()

        for row in ret["results"]["bindings"]:
            parent_label = row["cell_type"]["value"]    # The matched "parent" label
            child_label  = row["child_label"]["value"]  # A multi-level descendant
            if parent_label not in desc_map:
                desc_map[parent_label] = []
            desc_map[parent_label].append(child_label)

    return desc_map


###################################
#   MAIN EVALUATION LOGIC
###################################

def process_ndjson(input_ndjson: str):
    """
    Reads NDJSON lines that include:
      soma_joinid, ground_truth_label_clean, predicted_label_clean

    Gathers unique ground-truth and predicted labels -> queries Ubergraph with rdfs:subClassOf+
    -> compares row by row.

    Returns (results_list, metrics_dict).
    """
    row_data = []
    gt_labels = set()
    pr_labels = set()

    # 1) Read NDJSON & gather labels
    with open(input_ndjson, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cell_name = record.get("soma_joinid", None)
            # We'll cast it to string for uniformity
            if cell_name is not None:
                cell_name = str(cell_name)

            gt_label = record.get("ground_truth_label_clean", "").strip()
            pr_label = record.get("predicted_label_clean", "").strip()

            if gt_label:
                gt_labels.add(gt_label)
            if pr_label:
                pr_labels.add(pr_label)

            row_data.append({
                "cell_name": cell_name,
                "gt_label": gt_label,
                "pr_label": pr_label
            })

    # 2) Query *all* descendants for each label
    gt_desc_map = retrieve_all_descendants_from_ubergraph(list(gt_labels))
    pr_desc_map = retrieve_all_descendants_from_ubergraph(list(pr_labels))

    # 3) Compare row-by-row (scTab style, but multi-level)
    results = []
    for row in row_data:
        cell_name = row["cell_name"]
        gt_label  = row["gt_label"]
        pr_label  = row["pr_label"]

        # Descendants sets
        gt_descendants = set(gt_desc_map.get(gt_label, []))
        pr_descendants = set(pr_desc_map.get(pr_label, []))

        # scTab logic, extended for multi-level:
        #   TRUE if same label or predicted is in gt's descendant set
        #   FALSE if predicted is a parent of gt (i.e. gt is in predicted's descendant set)
        #   else "NO_DIRECT_CHILD_RELATION"
        if gt_label == pr_label and gt_label != "":
            verdict = "SAME_LABEL"
            match_scTab = "TRUE"
        elif pr_label in gt_descendants:
            # predicted is ANY-level child of ground_truth
            verdict = "PREDICTED_IS_CHILD_OF_GT"
            match_scTab = "TRUE"
        elif gt_label in pr_descendants:
            # ground_truth is ANY-level child of predicted
            verdict = "GT_IS_CHILD_OF_PREDICTED"
            match_scTab = "FALSE"
        else:
            verdict = "NO_DIRECT_CHILD_RELATION"
            match_scTab = "FALSE"

        results.append({
            "cell_name": cell_name,
            "ground_truth_label": gt_label,
            "predicted_label": pr_label,
            "verdict": verdict,
            "match_scTab": match_scTab,
            "ground_truth_children": sorted(gt_descendants),
            "predicted_children": sorted(pr_descendants)
        })

    # 4) Compute metrics
    num_cells = len(results)
    num_true = sum(1 for r in results if r["match_scTab"] == "TRUE")
    accuracy = float(num_true) / num_cells if num_cells else 0.0

    metrics = {
        "num_cells": num_cells,
        "num_true": num_true,
        "accuracy": accuracy
    }
    return results, metrics


def main():
    parser = argparse.ArgumentParser(
        description="scTab-like evaluation on NDJSON, but using rdfs:subClassOf+ (multi-level) in Ubergraph."
    )
    parser.add_argument("--input_ndjson", required=True,
                        help="Path to NDJSON produced by ols_label_normalize_ndjson.py.")
    parser.add_argument("--output_csv", required=False, default=None,
                        help="Path to a CSV summarizing row-by-row and final metrics (optional).")
    parser.add_argument("--output_json", required=False, default=None,
                        help="Path to a JSON with row-by-row + metrics (optional).")
    args = parser.parse_args()

    # 1) Process NDJSON
    all_rows, metrics = process_ndjson(args.input_ndjson)

    # 2) If requested, output CSV
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.writer(out_f)
            writer.writerow([
                "cell_name",
                "ground_truth_label",
                "predicted_label",
                "verdict",
                "match_scTab"
            ])
            for row in all_rows:
                writer.writerow([
                    row["cell_name"],
                    row["ground_truth_label"],
                    row["predicted_label"],
                    row["verdict"],
                    row["match_scTab"]
                ])
            # Final metrics
            writer.writerow([])
            writer.writerow(["EVALUATION_METRICS", "TOTAL_CELLS", metrics["num_cells"]])
            writer.writerow(["EVALUATION_METRICS", "NUM_TRUE", metrics["num_true"]])
            writer.writerow(["EVALUATION_METRICS", "ACCURACY", f"{metrics['accuracy']:.4f}"])

        print(f"CSV output saved to: {args.output_csv}")

    # 3) If requested, output JSON
    if args.output_json:
        data_out = {
            "evaluation_metrics": metrics,
            "cells": all_rows
        }
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump(data_out, jf, indent=2)
        print(f"JSON output saved to: {args.output_json}")

    # If neither CSV nor JSON was requested, we just do the evaluation in memory
    if not args.output_csv and not args.output_json:
        print("Evaluation done in memory (multi-level). Use --output_csv or --output_json to save results.")


if __name__ == "__main__":
    main()