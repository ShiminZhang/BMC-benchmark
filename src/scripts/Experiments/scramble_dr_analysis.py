"""
Scramble Regression Analysis Script

For every original instance that has scrambled solving-time data (produced by
prepare_formulas.py --scramble + collect_solving_time.py), this runs
direct_regression_analysis's model fitting on the original data and on each
scramble type's data, then summarizes how the best-fit model family (linear /
polynomial / exponential) shifts after scrambling.
"""

import os
import csv
import json
import glob
import argparse
from typing import Any, Dict, List, Optional, Tuple

from paths import (
    get_solving_times_dir,
    get_solving_times_path,
    get_scramble_dr_analysis_path,
    get_scramble_dr_summary_path,
)
from scramble_utils import SCRAMBLE_TYPES, make_scrambled_name, parse_scrambled_name
from direct_regression_analysis import analyze_instance, make_json_serializable

UNRELIABLE_R2_THRESHOLD = 0.5


def get_original_names_with_solving_times() -> List[str]:
    names = []
    for p in glob.glob(os.path.join(get_solving_times_dir(), "*.json")):
        name = os.path.splitext(os.path.basename(p))[0]
        if parse_scrambled_name(name) is None:
            names.append(name)
    return sorted(names)


def extract_family(analysis_result: Dict[str, Any]) -> Tuple[str, Optional[float]]:
    if "error" in analysis_result:
        return "error", None
    best_model = analysis_result.get("best_model", "None")
    best_r2 = analysis_result.get("best_r2_score")
    if best_model in (None, "None", "unknown") or best_r2 is None or best_r2 < UNRELIABLE_R2_THRESHOLD:
        return "unknown", best_r2
    return best_model, best_r2


def analyze_name_with_scrambles(name: str, seed: int, save_plot: bool) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"instance_name": name, "seed": seed}

    original_raw = analyze_instance(name, save_plot)
    original_family, original_r2 = extract_family(original_raw)
    entry["original"] = {"best_model": original_family, "best_r2_score": original_r2}

    for scramble_type in SCRAMBLE_TYPES:
        scrambled_name = make_scrambled_name(name, scramble_type, seed)
        if not os.path.exists(get_solving_times_path(scrambled_name)):
            entry[scramble_type] = None
            continue
        scrambled_raw = analyze_instance(scrambled_name, save_plot)
        family, r2 = extract_family(scrambled_raw)
        entry[scramble_type] = {"best_model": family, "best_r2_score": r2}

    return entry


def summarize(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for scramble_type in SCRAMBLE_TYPES:
        transitions: Dict[str, int] = {}
        unchanged = 0
        linear_to_nonlinear = 0
        nonlinear_to_linear = 0
        compared = 0
        for entry in entries:
            scrambled = entry.get(scramble_type)
            if scrambled is None:
                continue
            orig_family = entry["original"]["best_model"]
            scr_family = scrambled["best_model"]
            if orig_family == "error" or scr_family == "error":
                continue
            compared += 1
            key = f"{orig_family}->{scr_family}"
            transitions[key] = transitions.get(key, 0) + 1
            if orig_family == scr_family:
                unchanged += 1
            elif orig_family == "linear" and scr_family != "linear":
                linear_to_nonlinear += 1
            elif orig_family != "linear" and scr_family == "linear":
                nonlinear_to_linear += 1
        summary[scramble_type] = {
            "compared": compared,
            "unchanged": unchanged,
            "linear_to_nonlinear": linear_to_nonlinear,
            "nonlinear_to_linear": nonlinear_to_linear,
            "transitions": transitions,
        }
    return summary


def write_summary_csv(summary: Dict[str, Any], csv_path: str) -> None:
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scramble_type", "compared", "unchanged", "linear_to_nonlinear", "nonlinear_to_linear"])
        for scramble_type, info in summary.items():
            writer.writerow([
                scramble_type,
                info["compared"],
                info["unchanged"],
                info["linear_to_nonlinear"],
                info["nonlinear_to_linear"],
            ])


def main():
    parser = argparse.ArgumentParser(description="Compare regression model family before/after CNF scrambling")
    parser.add_argument("--seed", type=int, required=True, help="seed used with prepare_formulas.py --scramble")
    parser.add_argument("--name", type=str, help="analyze a single instance; default analyzes every instance with solving_times data")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output", type=str, help="per-instance results json (default: results/scramble_dr_analysis/scramble_dr_seed<seed>.json)")
    parser.add_argument("--summary-output", type=str, help="summary json (default: results/scramble_dr_analysis/scramble_dr_seed<seed>_summary.json)")
    args = parser.parse_args()

    save_plot = not args.no_plot
    names = [args.name] if args.name else get_original_names_with_solving_times()

    entries = []
    for name in names:
        print(f"Analyzing {name} (seed={args.seed}) ...")
        entries.append(analyze_name_with_scrambles(name, args.seed, save_plot))

    output_path = args.output or get_scramble_dr_analysis_path(args.seed)
    with open(output_path, "w") as f:
        json.dump(make_json_serializable(entries), f, indent=2)
    print(f"Per-instance results saved to: {output_path}")

    summary = summarize(entries)
    summary_path = args.summary_output or get_scramble_dr_summary_path(args.seed)
    with open(summary_path, "w") as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    print(f"Summary saved to: {summary_path}")

    csv_path = summary_path[:-len(".json")] + ".csv" if summary_path.endswith(".json") else summary_path + ".csv"
    write_summary_csv(summary, csv_path)
    print(f"CSV summary saved to: {csv_path}")


if __name__ == "__main__":
    main()
