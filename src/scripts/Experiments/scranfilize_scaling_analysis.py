"""
Scranfilize Scaling Analysis for BMCBenchmark.

This mirrors the existing scramble regression workflow, but uses the repository
root's External/scranfilize binary and named profiles. It produces virtual BMC
instance names of the form:

    <name>__scran_<profile>_s<seed>

The generated CNFs and solver logs then flow through the existing
collect_solving_time.py and direct_regression_analysis.py scaling analysis.
"""

import argparse
import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

from GenericRA import GetDataFromLog, LOG, TOGGLE_SHOWLOG
from direct_regression_analysis import analyze_instance, make_json_serializable
from paths import (
    get_cnf_path,
    get_solving_log_dir,
    get_solving_log_path,
    get_solving_times_path,
)
from prepare_formulas import generate_cnf
from scramble_dr_analysis import extract_family


SCRANFILIZE_PROFILES: Dict[str, List[str]] = {
    "clause_light": ["-f", "0", "-v", "0", "-c", "0.01"],
    "clause_mid": ["-f", "0", "-v", "0", "-c", "0.10"],
    "clause_full": ["-f", "0", "-v", "0", "-P"],
    "var_light": ["-f", "0", "-v", "0.01", "-c", "0"],
    "var_full": ["-f", "0", "-p", "-c", "0"],
    "all_light": ["-f", "0", "-v", "0.01", "-c", "0.01"],
}

_SCRAN_NAME_RE = re.compile(r"^(?P<name>.+)__scran_(?P<profile>[a-zA-Z0-9_]+)_s(?P<seed>-?\d+)$")


def make_scranfilized_name(name: str, profile: str, seed: int) -> str:
    return f"{name}__scran_{profile}_s{seed}"


def parse_scranfilized_name(virtual_name: str) -> Optional[Tuple[str, str, int]]:
    match = _SCRAN_NAME_RE.match(virtual_name)
    if not match:
        return None
    return match.group("name"), match.group("profile"), int(match.group("seed"))


def repo_root() -> str:
    current = os.path.abspath(os.path.dirname(__file__))
    while True:
        if os.path.isdir(os.path.join(current, "External", "scranfilize")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("could not locate ProofDoorTools root containing External/scranfilize")
        current = parent


def find_scranfilize_binary(explicit: Optional[str] = None) -> str:
    candidates = []
    if explicit:
        candidates.append(explicit)
    env_path = os.environ.get("SCRANFILIZE")
    if env_path:
        candidates.append(env_path)
    candidates.append(os.path.join(repo_root(), "External", "scranfilize", "scranfilize"))
    for path in candidates:
        if path and os.path.exists(path) and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "scranfilize binary not found. From ProofDoorTools root, run ./build_dependencies.sh "
        "or set SCRANFILIZE=/path/to/scranfilize."
    )


def load_original_ks(name: str, k_limit: int = 0) -> List[int]:
    path = get_solving_times_path(name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"solving time json not found for {name}: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    ks = sorted(int(k) for k in data.keys())
    if k_limit > 0:
        ks = [k for k in ks if k <= k_limit]
    return ks


def run_formula(name: str, K: int, solver: str, time_limit: int) -> bool:
    cnf_path = get_cnf_path(name, K)
    solving_log_path = get_solving_log_path(name, K)
    LOG(f"Running formula {name} with K={K} up to limit {time_limit}")
    try:
        result = subprocess.run(
            f"{solver} {cnf_path} --plain -t {time_limit}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=time_limit,
        )
        with open(solving_log_path, "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write(f"\nSTDERR:\n{result.stderr}")
        return True
    except subprocess.TimeoutExpired:
        LOG(f"Formula {name} with K={K} timed out after {time_limit} seconds")
        return False


def generate_scranfilized_cnf(
    name: str,
    K: int,
    profile: str,
    seed: int,
    scranfilize_binary: Optional[str],
) -> str:
    if profile not in SCRANFILIZE_PROFILES:
        raise ValueError(f"unknown profile {profile}; choices: {sorted(SCRANFILIZE_PROFILES)}")
    if not generate_cnf(name, K):
        raise FileNotFoundError(f"failed to generate original CNF for {name}.{K}")
    original = get_cnf_path(name, K)
    virtual_name = make_scranfilized_name(name, profile, seed)
    output = get_cnf_path(virtual_name, K)
    binary = find_scranfilize_binary(scranfilize_binary)
    cmd = [binary, "-s", str(seed), *SCRANFILIZE_PROFILES[profile], original, output, "--force"]
    subprocess.run(cmd, check=True)
    return output


def generate_run_series(
    name: str,
    profile: str,
    seed: int,
    time_limit: int,
    k_limit: int,
    scranfilize_binary: Optional[str],
) -> str:
    virtual_name = make_scranfilized_name(name, profile, seed)
    solver = "./solvers/bin/cadical"
    for K in load_original_ks(name, k_limit):
        generate_scranfilized_cnf(name, K, profile, seed, scranfilize_binary)
        if not run_formula(virtual_name, K, solver, time_limit):
            break
    return virtual_name


def collect_scranfilized_solving_time(virtual_name: str, include_nvar: bool = False) -> Optional[str]:
    formula_dir = os.path.join(get_solving_log_dir(), virtual_name)
    if not os.path.isdir(formula_dir):
        LOG(f"formula_dir {formula_dir} does not exist, skipping")
        return None
    output: Dict[str, Dict[str, Any]] = {}
    for log_file in os.listdir(formula_dir):
        if not log_file.endswith(".log"):
            continue
        parts = log_file.split(".")
        if len(parts) != 3:
            continue
        try:
            K = int(parts[1])
        except ValueError:
            continue
        cnf_path = get_cnf_path(virtual_name, K)
        if not os.path.exists(cnf_path):
            continue
        with open(cnf_path, "r") as f:
            size_of_cnf = len(f.readlines()) - 2
        solving_time, is_unsat = GetDataFromLog(os.path.join(formula_dir, log_file))
        if solving_time is None:
            continue
        record: Dict[str, Any] = {
            "solving_time": solving_time,
            "size_of_cnf": size_of_cnf,
            "is_unsat": is_unsat,
        }
        if include_nvar:
            with open(cnf_path, "r") as f:
                for line in f:
                    if line.startswith("p cnf"):
                        record["nvar"] = int(line.split()[2])
                        break
        output[str(K)] = record
    output = dict(sorted(output.items(), key=lambda kv: int(kv[0])))
    out_path = get_solving_times_path(virtual_name)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=4)
    return out_path


def analyze_variant(name: str, profile: str, seed: int, save_plot: bool) -> Dict[str, Any]:
    virtual_name = make_scranfilized_name(name, profile, seed)
    original_raw = analyze_instance(name, save_plot)
    variant_raw = analyze_instance(virtual_name, save_plot)
    original_family, original_r2 = extract_family(original_raw)
    variant_family, variant_r2 = extract_family(variant_raw)
    return {
        "instance_name": name,
        "variant_name": virtual_name,
        "profile": profile,
        "seed": seed,
        "original": {"best_model": original_family, "best_r2_score": original_r2},
        "scranfilized": {"best_model": variant_family, "best_r2_score": variant_r2},
    }


def main() -> None:
    TOGGLE_SHOWLOG(True)
    parser = argparse.ArgumentParser(description="Generate, collect, and analyze scranfilized BMC scaling.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile", choices=sorted(SCRANFILIZE_PROFILES), default="clause_mid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_limit", type=int, default=1600)
    parser.add_argument("--k_limit", type=int, default=0)
    parser.add_argument("--scranfilize_binary", default=None)
    parser.add_argument("--generate-run", action="store_true", default=False)
    parser.add_argument("--collect", action="store_true", default=False)
    parser.add_argument("--analyze", action="store_true", default=False)
    parser.add_argument("--all", action="store_true", default=False)
    parser.add_argument("--include_nvar", action="store_true", default=False)
    parser.add_argument("--no-plot", action="store_true", default=False)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.all or args.generate_run:
        generate_run_series(
            args.name,
            args.profile,
            args.seed,
            args.time_limit,
            args.k_limit,
            args.scranfilize_binary,
        )

    virtual_name = make_scranfilized_name(args.name, args.profile, args.seed)
    if args.all or args.collect:
        out_path = collect_scranfilized_solving_time(virtual_name, include_nvar=args.include_nvar)
        print(f"Collected solving times: {out_path}")

    if args.all or args.analyze:
        result = analyze_variant(args.name, args.profile, args.seed, save_plot=not args.no_plot)
        output = args.output or f"./results/scranfilize_scaling/{virtual_name}.json"
        os.makedirs(os.path.dirname(output), exist_ok=True)
        with open(output, "w") as f:
            json.dump(make_json_serializable(result), f, indent=2)
        print(f"Analysis saved to: {output}")


if __name__ == "__main__":
    main()
