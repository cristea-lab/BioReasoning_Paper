#!/usr/bin/env python3
"""
Post-process results.csv:
  • drop rows whose ground-truth label is blank
  • compute accuracy / macro-F1 / weighted-F1 / median-F1
  • add percentile-bootstrap 95 % CIs (accuracy, macro-F1)
  • add Wilson 95 % CI for accuracy               
  • write cleaned per-cell block + metrics block

Usage
-----
python d_remove_empty_match_and_bootstrap.py \
    --input_csv  results.csv \
    --output_csv results_cleaned_v2.csv \
    [--n_boot 10000] [--alpha 0.05] [--seed 0]
"""
# ---------------------------------------------------------------------------
import argparse, csv
from typing import List, Dict
from statistics import median
import numpy as np
from statsmodels.stats.proportion import proportion_confint   
# ---------------------------------------------------------------------------


# --------------------------- metric helpers --------------------------------
def _compute_metrics(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return dict(accuracy=0.0, macro_f1=0.0,
                    weighted_f1=0.0, median_f1=0.0, num_true=0)

    gt_labels = {r["gt"] for r in rows}
    stats = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in gt_labels}
    num_true = 0

    for r in rows:
        gt, ev, ok = r["gt"], r["eval_pr"], r["ok"]
        if ok:
            num_true += 1
            stats[gt]["tp"] += 1
        else:
            stats[gt]["fn"] += 1
            if ev in stats:
                stats[ev]["fp"] += 1

    accuracy = num_true / len(rows)

    f1_vals, macro_sum, weighted_sum, total_support = [], 0.0, 0.0, 0
    for lbl, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec  = tp / (tp + fn) if tp + fn else 0.0
        f1   = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn
        f1_vals.append(f1)
        macro_sum    += f1
        weighted_sum += f1 * support
        total_support+= support

    macro_f1    = macro_sum / len(stats)
    weighted_f1 = weighted_sum / total_support if total_support else 0.0
    median_f1   = median(f1_vals)

    return dict(accuracy=accuracy,
                macro_f1=macro_f1,
                weighted_f1=weighted_f1,
                median_f1=median_f1,
                num_true=num_true)


def _bootstrap(rows: List[Dict[str, str]],
               n_boot: int,
               alpha: float,
               seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n   = len(rows)
    acc = np.empty(n_boot)
    mac = np.empty(n_boot)

    for i in range(n_boot):
        idx   = rng.integers(0, n, size=n)
        samp  = [rows[j] for j in idx]
        m     = _compute_metrics(samp)
        acc[i]= m["accuracy"]
        mac[i]= m["macro_f1"]

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return dict(ACCURACY_CI_LO_BOOT = float(np.percentile(acc, lo)),
                ACCURACY_CI_HI_BOOT = float(np.percentile(acc, hi)),
                MACRO_F1_CI_LO_BOOT = float(np.percentile(mac, lo)),
                MACRO_F1_CI_HI_BOOT = float(np.percentile(mac, hi)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv",  required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--alpha",  type=float, default=0.05)
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    # ------------------- read & filter rows -----------------------------
    cell_rows: List[List[str]] = []
    with open(args.input_csv, newline='', encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)                     # skip header
        for row in reader:
            if not row:
                continue
            if row[0] == "EVALUATION_METRICS":
                break
            cell_rows.append(row)

    kept: List[Dict[str, str]] = []
    for cell, gt, pr, verdict, ok in cell_rows:
        if not gt.strip():
            continue
        ok_bool = (ok.upper() == "TRUE")
        eval_pr = gt if ok_bool else pr
        kept.append({"cell": cell, "gt": gt, "pr": pr,
                     "verdict": verdict, "ok": ok_bool,
                     "eval_pr": eval_pr})

    # --------------- metrics + bootstrap + Wilson ----------------------
    base = _compute_metrics(kept)
    ci_b = _bootstrap(kept, args.n_boot, args.alpha, args.seed)

    lo_w, hi_w = proportion_confint(         
        count=base["num_true"],
        nobs=len(kept),
        alpha=args.alpha,
        method="wilson"
    )
    ci_w = dict(ACCURACY_CI_LO_WILSON=float(lo_w),
                ACCURACY_CI_HI_WILSON=float(hi_w))

    # ------------------- write cleaned CSV ------------------------------
    with open(args.output_csv, 'w', newline='', encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell","ground_truth_label","predicted_label",
                    "verdict","match_scTab"])
        for r in kept:
            w.writerow([r["cell"], r["gt"], r["pr"], r["verdict"],
                        "TRUE" if r["ok"] else "FALSE"])
        w.writerow([])

        w.writerow(["EVALUATION_METRICS","TOTAL_CELLS", len(kept)])
        w.writerow(["EVALUATION_METRICS","NUM_TRUE",    base["num_true"]])
        w.writerow(["EVALUATION_METRICS","ACCURACY",    f"{base['accuracy']:.4f}"])
        w.writerow(["EVALUATION_METRICS","MACRO_F1",    f"{base['macro_f1']:.4f}"])
        w.writerow(["EVALUATION_METRICS","WEIGHTED_F1", f"{base['weighted_f1']:.4f}"])
        w.writerow(["EVALUATION_METRICS","MEDIAN_F1",   f"{base['median_f1']:.4f}"])

        # bootstrap CIs
        for k in ["ACCURACY_CI_LO_BOOT","ACCURACY_CI_HI_BOOT",
                  "MACRO_F1_CI_LO_BOOT","MACRO_F1_CI_HI_BOOT"]:
            w.writerow(["EVALUATION_METRICS", k, f"{ci_b[k]:.4f}"])
        # Wilson CI rows 
        for k in ["ACCURACY_CI_LO_WILSON","ACCURACY_CI_HI_WILSON"]:
            w.writerow(["EVALUATION_METRICS", k, f"{ci_w[k]:.4f}"])

    print(f"Cleaned CSV with bootstrap + Wilson CIs written to {args.output_csv}")


if __name__ == "__main__":
    main()