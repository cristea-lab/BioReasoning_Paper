#!/usr/bin/env python3
"""
scTab-style ontology-aware evaluation  (v3 — **CSV-only, metrics-only**)
-----------------------------------------------------------------------
Changes relative to v2
~~~~~~~~~~~~~~~~~~~~~
1. **Keeps rows even if the prediction is empty**; drops rows only when the
   *ground-truth* label is missing/blank.
2. **Writes exactly one CSV file** whose rows match the required three-column
   format, e.g.::

        EVALUATION_METRICS,TOTAL_CELLS,10000
        EVALUATION_METRICS,NUM_TRUE,2284
        ...

   There is **no JSON output path**, and the per-cell records are gone.

Reported metrics (unchanged):
    - accuracy          (overall row correctness)
    - macro_f1          (un-weighted mean of per-label F1)
    - median_f1         (median of per-label F1s)
    - weighted_f1       (support-weighted mean of per-label F1)
"""

# ---------------------------------------------------------------------------
import argparse, csv, json
from typing import Dict, List
from statistics import median
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON
# ---------------------------------------------------------------------------

def batched(seq, n):
    """Yield successive *n*-sized chunks from *seq*."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def descendants_query(batch: List[str]) -> str:
    values = " ".join(f'"{escape_literal(lbl)}"' for lbl in batch)
    return (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX owl:  <http://www.w3.org/2002/07/owl#>\n"
        "SELECT * WHERE {\n"
        "  ?parent rdfs:label ?lab.\n"
        "  BIND( str(?lab) AS ?lab_str )\n"
        "  ?child  rdfs:subClassOf+ ?parent; rdfs:label ?child_label.\n"
        "  ?parent rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>.\n"
        "  ?child  rdfs:isDefinedBy <http://purl.obolibrary.org/obo/cl.owl>.\n"
        f"  VALUES ?lab_str {{ {values} }}\n"
        "}"
    )


def get_descendants(labels: List[str]) -> Dict[str, List[str]]:
    """Return a mapping *parent_label ➜ list[descendant_label]* via Ubergraph."""
    if not labels:
        return {}

    desc: Dict[str, List[str]] = {}
    sparql = SPARQLWrapper("https://ubergraph.apps.renci.org/sparql")
    sparql.method = "POST"
    sparql.setReturnFormat(SPARQLJSON)

    for batch in batched(labels, 80):
        sparql.setQuery(descendants_query(batch))
        try:
            res = sparql.queryAndConvert()
        except Exception:
            continue  # silently skip network errors

        for row in res["results"]["bindings"]:
            parent = row["lab_str"]["value"]
            child = row["child_label"]["value"]
            desc.setdefault(parent, []).append(child)

    return desc

# ---------------------------------------------------------------------------

def evaluate(ndjson_path: str):
    """Run the ontology-aware evaluation and return the metrics dict."""
    rows, gt_labels, pr_labels = [], set(), set()

    with open(ndjson_path, encoding="utf-8") as f:
        for ln in f:
            if not (ln := ln.strip()):
                continue
            obj = json.loads(ln)
            gt = str(obj.get("ground_truth_label_clean", "")).strip()
            pr = str(obj.get("predicted_label_clean", "")).strip()

            # --- Drop if GT label is missing/blank ----------------------
            if not gt:
                continue

            rows.append({"gt": gt, "pr": pr})
            gt_labels.add(gt)
            if pr:
                pr_labels.add(pr)

    if not rows:  # all rows removed ➜ zero metrics
        return {
            "TOTAL_CELLS": 0,
            "NUM_TRUE": 0,
            "ACCURACY": 0.0,
            "MACRO_F1": 0.0,
            "WEIGHTED_F1": 0.0,
            "MEDIAN_F1": 0.0,
        }

    # -------- ontology look-ups ----------------------------------------
    gt_desc = get_descendants(list(gt_labels))
    pr_desc = get_descendants(list(pr_labels))

    # -------- verdicts & collapse -------------------------------------
    eval_rows: List[Dict[str, str]] = []
    true_rows = 0

    for r in rows:
        gt, pr = r["gt"], r["pr"]
        ok = False

        if pr and gt == pr:
            ok = True
        elif pr and pr in gt_desc.get(gt, []):
            ok = True
        # If prediction is blank or doesn't match, ok remains False

        eval_pr = gt if (ok and pr and pr != gt) else pr
        eval_rows.append({"gt": gt, "eval_pr": eval_pr, "ok": ok})
        true_rows += int(ok)

    accuracy = true_rows / len(eval_rows)

    # -------- confusion matrix (GT labels only) ------------------------
    stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in gt_labels}

    for r in eval_rows:
        gt, ev, ok = r["gt"], r["eval_pr"], r["ok"]
        if ok:
            stats[gt]["tp"] += 1
        else:
            stats[gt]["fn"] += 1
            if ev in stats:
                stats[ev]["fp"] += 1

    # -------- per-label metrics ---------------------------------------
    f1_vals, macro_sum, weighted_sum, total_support = [], 0.0, 0.0, 0

    for lbl, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn

        macro_sum += f1
        weighted_sum += f1 * support
        total_support += support
        f1_vals.append(f1)

    macro_f1 = macro_sum / len(stats)
    weighted_f1 = weighted_sum / total_support if total_support else 0.0
    median_f1 = median(f1_vals) if f1_vals else 0.0

    return {
        "TOTAL_CELLS": len(eval_rows),
        "NUM_TRUE": true_rows,
        "ACCURACY": accuracy,
        "MACRO_F1": macro_f1,
        "WEIGHTED_F1": weighted_f1,
        "MEDIAN_F1": median_f1,
    }

# ---------------------------------------------------------------------------

def write_metrics_csv(path: str, metrics: Dict[str, float]):
    """Write the metrics in the required three-column CSV format."""
    order = [
        ("TOTAL_CELLS", int),
        ("NUM_TRUE", int),
        ("ACCURACY", float),
        ("MACRO_F1", float),
        ("WEIGHTED_F1", float),
        ("MEDIAN_F1", float),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for key, _ in order:
            val = metrics[key]
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            w.writerow(["EVALUATION_METRICS", key, val_str])

# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="scTab ontology evaluation ➜ CSV metrics")
    ap.add_argument("--input_ndjson", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    metrics = evaluate(args.input_ndjson)
    write_metrics_csv(args.output_csv, metrics)
    print(f"Metrics CSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
