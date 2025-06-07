#!/usr/bin/env python3
"""
scTab-style ontology-aware evaluation
-------------------------------------
* TRUE if prediction == ground truth OR prediction is any descendant.
* For TRUE rows where prediction is a descendant, collapse the prediction
  up to the parent label before building the confusion matrix.
* Per-label TP/FP/FN are counted **only** for labels that appear as ground
  truth.  Metrics reported:
    - accuracy           (overall row correctness)
    - macro_f1           (un-weighted mean of per-label F1)
    - median_f1          (median of per-label F1s)
    - weighted_f1        (support-weighted mean of per-label F1)
"""

# ---------------------------------------------------------------------------
import argparse, csv, json
from typing import Dict, List
from statistics import median
from SPARQLWrapper import SPARQLWrapper, JSON as SPARQLJSON
# ---------------------------------------------------------------------------

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]

def escape_literal(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')

def descendants_query(batch: List[str]) -> str:
    values = " ".join(f"\"{escape_literal(lbl)}\"" for lbl in batch)
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
    if not labels: return {}
    desc: Dict[str, List[str]] = {}
    sparql = SPARQLWrapper("https://ubergraph.apps.renci.org/sparql")
    sparql.method = "POST"
    sparql.setReturnFormat(SPARQLJSON)

    for batch in batched(labels, 80):
        sparql.setQuery(descendants_query(batch))
        try:                 res = sparql.queryAndConvert()
        except Exception:    continue
        for row in res["results"]["bindings"]:
            parent = row["lab_str"]["value"]
            child  = row["child_label"]["value"]
            desc.setdefault(parent, []).append(child)
    return desc
# ---------------------------------------------------------------------------

def evaluate(ndjson_path: str):
    # -------- load data ----------------------------------------------------
    rows, gt_labels, pr_labels = [], set(), set()
    with open(ndjson_path, encoding="utf-8") as f:
        for ln in f:
            if not (ln := ln.strip()): continue
            obj = json.loads(ln)
            gt = obj.get("ground_truth_label_clean", "").strip()
            pr = obj.get("predicted_label_clean", "").strip()
            rows.append({"cell": str(obj.get("soma_joinid", "")), "gt": gt, "pr": pr})
            if gt: gt_labels.add(gt)
            if pr: pr_labels.add(pr)

    # -------- ontology look-ups -------------------------------------------
    gt_desc = get_descendants(list(gt_labels))
    pr_desc = get_descendants(list(pr_labels))

    # -------- verdicts + collapse -----------------------------------------
    eval_rows, true_rows = [], 0
    for r in rows:
        gt, pr = r["gt"], r["pr"]
        ok = False
        if gt and pr and gt == pr:
            verdict, ok = "SAME_LABEL", True
        elif pr and pr in gt_desc.get(gt, []):
            verdict, ok = "PREDICTED_IS_CHILD_OF_GT", True
        elif gt and gt in pr_desc.get(pr, []):
            verdict, ok = "GT_IS_CHILD_OF_PREDICTED", False
        else:
            verdict = "NO_DIRECT_CHILD_RELATION"

        eval_pr = gt if (ok and pr != gt) else pr
        eval_rows.append({"cell": r["cell"], "gt": gt, "pr": pr,
                          "eval_pr": eval_pr, "verdict": verdict,
                          "ok": "TRUE" if ok else "FALSE"})
        true_rows += int(ok)

    accuracy = true_rows / len(eval_rows) if eval_rows else 0.0

    # -------- confusion matrix (GT labels only) ---------------------------
    stats = {lbl: {"tp":0,"fp":0,"fn":0} for lbl in gt_labels}
    for r in eval_rows:
        gt, ev, ok = r["gt"], r["eval_pr"], (r["ok"] == "TRUE")
        if gt in stats:
            if ok: stats[gt]["tp"] += 1
            else:  stats[gt]["fn"] += 1
        if (not ok) and (ev in stats):
            stats[ev]["fp"] += 1

    # -------- per-label metrics & weighted sums ---------------------------
    f1_vals, macro_sum, weighted_sum, total_support = [], 0.0, 0.0, 0
    for lbl, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2*prec*rec/(prec+rec) if prec+rec else 0.0
        support = tp + fn                 # rows whose GT label = lbl
        s.update({"precision":prec, "recall":rec, "f1":f1, "support":support})

        macro_sum    += f1
        weighted_sum += f1 * support
        total_support+= support
        f1_vals.append(f1)

    macro_f1   = macro_sum / len(stats) if stats else 0.0
    weighted_f1= weighted_sum / total_support if total_support else 0.0
    median_f1  = median(f1_vals) if f1_vals else 0.0

    metrics = {
        "num_cells":  len(eval_rows),
        "num_true":   true_rows,
        "accuracy":   accuracy,
        "macro_f1":   macro_f1,
        "weighted_f1":weighted_f1,
        "median_f1":  median_f1
    }
    return eval_rows, metrics
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="scTab-style ontology-aware evaluation")
    ap.add_argument("--input_ndjson", required=True)
    ap.add_argument("--output_csv")
    ap.add_argument("--output_json")
    args = ap.parse_args()

    rows, metrics = evaluate(args.input_ndjson)

    # -------- CSV ----------------------------------------------------------
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["cell","ground_truth_label","predicted_label",
                        "verdict","match_scTab"])
            for r in rows:
                w.writerow([r["cell"], r["gt"], r["pr"], r["verdict"], r["ok"]])
            w.writerow([])
            w.writerow(["EVALUATION_METRICS","TOTAL_CELLS", metrics["num_cells"]])
            w.writerow(["EVALUATION_METRICS","NUM_TRUE",    metrics["num_true"]])
            w.writerow(["EVALUATION_METRICS","ACCURACY",    f"{metrics['accuracy']:.4f}"])
            w.writerow(["EVALUATION_METRICS","MACRO_F1",    f"{metrics['macro_f1']:.4f}"])
            w.writerow(["EVALUATION_METRICS","WEIGHTED_F1", f"{metrics['weighted_f1']:.4f}"])
            w.writerow(["EVALUATION_METRICS","MEDIAN_F1",   f"{metrics['median_f1']:.4f}"])
        print(f"CSV written to {args.output_csv}")

    # -------- JSON ---------------------------------------------------------
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as jf:
            json.dump({"evaluation_metrics":metrics,"cells":rows}, jf, indent=2)
        print(f"JSON written to {args.output_json}")

    if not args.output_csv and not args.output_json:
        print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()