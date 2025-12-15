import json
import argparse
import os
import csv
from typing import List, Tuple, Optional, Dict, Any
from paths import get_solving_times_path


def load_solving_series(instance_name: str) -> Tuple[List[int], List[float]]:
    """
    Load solving_time series for an instance, returning sorted k and corresponding times.
    """
    if not instance_name:
        return [], []
    path = get_solving_times_path(instance_name)
    if not os.path.exists(path):
        # Missing data file; return empty to trigger fallbacks
        return [], []
    with open(path, 'r') as f:
        data = json.load(f)
    # Keys are string ks; convert to ints and sort
    items = []
    for k_str, rec in data.items():
        try:
            k_val = int(k_str)
            t_val = float(rec.get('solving_time', 0.0))
            items.append((k_val, t_val))
        except Exception:
            continue
    items.sort(key=lambda x: x[0])
    ks = [k for k, _ in items]
    ts = [t for _, t in items]
    return ks, ts


def first_local_max_k(ks: List[int], ts: List[float]) -> Optional[int]:
    """
    Return the first k where ts[i-1] < ts[i] > ts[i+1].
    Uses immediate neighbors in the sorted-by-k order.
    """
    if len(ts) < 3:
        return None
    for i in range(1, len(ts) - 1):
        if ts[i] > ts[i - 1] and ts[i] > ts[i + 1]:
            return ks[i]
    return None


def first_k_with_time_gt(ks: List[int], ts: List[float], threshold: float) -> Optional[int]:
    """
    Return the first k where solving_time > threshold.
    """
    for k, t in zip(ks, ts):
        if t > threshold:
            return k
    return None


def compute_recommended_k(instance_name: str, best_model: str) -> int:
    """
    Compute recommended K based on best_model:
      - exponential: first local max; if none, first solving_time > 1; if none, -1
      - linear: 10
      - polynomial: -1
    """
    model = (best_model or "").strip().lower()
    if model == 'linear':
        return 10
    if model == 'polynomial':
        return -1
    # default treat as exponential-like
    ks, ts = load_solving_series(instance_name)
    k_local = first_local_max_k(ks, ts)
    if k_local is not None:
        return k_local
    k_gt1 = first_k_with_time_gt(ks, ts, 1.0)
    return k_gt1 if k_gt1 is not None else -1


def infer_best_model(entry: Dict[str, Any]) -> Optional[str]:
    """
    Try to infer best model from a summary entry.
    Supports compact format: {'best_model': 'linear', ...}
    Or raw results dict: {'results': {'linear': {'r2_score': ...}, ...}}
    """
    if isinstance(entry, dict):
        if 'best_model' in entry:
            return entry.get('best_model')
        if 'results' in entry and isinstance(entry['results'], dict):
            # choose by max r2_score
            best = None
            best_r2 = float('-inf')
            for name, info in entry['results'].items():
                try:
                    r2 = float(info.get('r2_score'))
                except Exception:
                    continue
                if r2 > best_r2:
                    best_r2 = r2
                    best = name
            return best
    return None


def add_local_max_k_to_summary(summary_file: str) -> None:
    """
    Add the recommended K to each instance in the summary file as 'local_max_k'.
    - exponential instances: first local max, else first time > 1, else -1
    - linear: 10
    - polynomial: -1
    """
    with open(summary_file, 'r') as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Summary file must contain a JSON object mapping instance_name to result info.")
    rows_for_csv = []
    for instance_name, entry in data.items():
        best_model = infer_best_model(entry) or ""
        k_val = compute_recommended_k(instance_name, best_model)
        entry['local_max_k'] = k_val
        rows_for_csv.append((instance_name, best_model, k_val))
    with open(summary_file, 'w') as f:
        json.dump(data, f, indent=2)
    # Also write a CSV with (instance_name, best_model, local_max_k)
    out_csv = summary_file[:-5] + '_k.csv' if summary_file.endswith('.json') else summary_file + '_k.csv'
    with open(out_csv, 'w', newline='') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['instance_name', 'best_model', 'local_max_k'])
        for instance_name, best_model, k_val in rows_for_csv:
            writer.writerow([instance_name, best_model, k_val])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', '-s', type=str, default='regression_summary.json',
                        help='Summary file for results (JSON)')
    args = parser.parse_args()
    add_local_max_k_to_summary(args.summary)


if __name__ == '__main__':
    main()