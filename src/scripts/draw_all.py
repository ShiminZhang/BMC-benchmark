#!/usr/bin/env python3
"""
Draw aggregated plots across all prepared solving-time JSONs.

Outputs two figures:
- Scatter cloud: aggregated point cloud (x configurable; y configurable)
- (Optional) Lines: disabled by default, enable with --with-lines

Usage:
  python -m src.scripts.draw_all
  python -m src.scripts.draw_all --min-family-points 1 --family-bins 40
  python -m src.scripts.draw_all --also-log
  python -m src.scripts.draw_all --k 10
  python -m src.scripts.draw_all --with-lines
  python -m src.scripts.draw_all --x n
  python -m src.scripts.draw_all --x n --y m
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(it, **_kwargs):  # type: ignore
        return it

# Ensure headless operation on clusters/CI
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .paths import get_plot_path, get_solving_times_dir


@dataclass(frozen=True)
class SolvingPoint:
    instance: str
    family: str
    size_of_cnf: float
    solving_time: float
    num_vars: float


def infer_family(instance_name: str) -> str:
    """
    Treat each instance name as its own family.

    (User requirement: one benchmark name == one family.)
    """
    base = instance_name.strip()
    return base if base else "unknown"


def iter_solving_time_files() -> Iterable[Tuple[str, str]]:
    """Yield (instance_name, json_path) for all prepared solving-times JSONs."""
    solving_dir = get_solving_times_dir()
    if not os.path.isdir(solving_dir):
        return
    for fn in sorted(os.listdir(solving_dir)):
        if not fn.endswith(".json"):
            continue
        instance = fn[:-5]
        yield instance, os.path.join(solving_dir, fn)


def load_all_solving_points() -> List[SolvingPoint]:
    points: List[SolvingPoint] = []
    files = list(iter_solving_time_files())
    for instance, path in tqdm(files, desc="Loading solving_times", unit="file"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        family = infer_family(instance)
        for _k, v in data.items():
            if not isinstance(v, dict):
                continue
            try:
                y = float(v.get("solving_time"))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):
                continue
            try:
                size_of_cnf = float(v.get("size_of_cnf"))
            except (TypeError, ValueError):
                size_of_cnf = float("nan")
            try:
                num_vars = float(v.get("nvar"))
            except (TypeError, ValueError):
                num_vars = float("nan")
            if not np.isfinite(num_vars):
                print(f"[warn] missing num_vars for instance={instance} k={_k}")
            if not np.isfinite(size_of_cnf) and not np.isfinite(num_vars):
                continue
            points.append(
                SolvingPoint(
                    instance=instance,
                    family=family,
                    size_of_cnf=size_of_cnf,
                    num_vars=num_vars,
                    solving_time=y,
                )
            )
    return points


def _sanitize_positive(vals: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Replace non-finite / non-positive values to allow log-scale plotting."""
    v2 = vals.astype(float, copy=True)
    v2[~np.isfinite(v2)] = np.nan
    v2[v2 <= 0] = eps
    return v2


def _trim_by_percentile(vals: np.ndarray, pct: float) -> Tuple[float, float]:
    """Return (lo, hi) percentile cutoffs for finite vals."""
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        return (np.nan, np.nan)
    lo = float(np.quantile(v, pct))
    hi = float(np.quantile(v, 1.0 - pct))
    return lo, hi


def _scatter_out_paths(out_path: str) -> Tuple[str, str, str, str]:
    """
    Return (base, trim_x, trim_y, trim_both) output paths.
    """
    root, ext = os.path.splitext(out_path)
    return (
        out_path,
        f"{root}_trim_x5{ext}",
        f"{root}_trim_y5{ext}",
        f"{root}_trim_both5{ext}",
    )


def _scatter_from_arrays(
    xs: np.ndarray,
    ys_raw: np.ndarray,
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    loglog: bool,
    point_size: float,
    alpha: float,
) -> None:
    # For plotting, sanitize y for log-scale (also makes 0 visible)
    ys = _sanitize_positive(ys_raw)
    plt.figure(figsize=(12, 8))
    plt.scatter(xs, ys, s=point_size, alpha=alpha, c="black", edgecolors="none")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_scatter_variants(
    xs: np.ndarray,
    ys_raw: np.ndarray,
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    loglog: bool,
    point_size: float,
    alpha: float,
) -> None:
    # Base mask: finite and x>0 (required for log-x if enabled)
    base_mask = np.isfinite(xs) & np.isfinite(ys_raw) & (xs > 0)
    xs0 = xs[base_mask]
    ys0 = ys_raw[base_mask]

    out_base, out_trim_x, out_trim_y, out_trim_both = _scatter_out_paths(out_path)

    print(f"{title} | points N={xs0.size} | out={out_base}")
    _scatter_from_arrays(
        xs0,
        ys0,
        out_base,
        title=title,
        x_label=x_label,
        y_label=y_label,
        loglog=loglog,
        point_size=point_size,
        alpha=alpha,
    )

    # Trim by x-axis percentiles
    x_lo, x_hi = _trim_by_percentile(xs0, 0.05)
    mx = None
    if np.isfinite(x_lo) and np.isfinite(x_hi) and x_lo <= x_hi:
        mx = (xs0 >= x_lo) & (xs0 <= x_hi)
        print(f"{title} [trim x 5%] | points N={int(np.sum(mx))} | out={out_trim_x}")
        _scatter_from_arrays(
            xs0[mx],
            ys0[mx],
            out_trim_x,
            title=f"{title} (trim x 5%)",
            x_label=x_label,
            y_label=y_label,
            loglog=loglog,
            point_size=point_size,
            alpha=alpha,
        )

    # Trim by y-axis percentiles (use raw y for trimming)
    y_lo, y_hi = _trim_by_percentile(ys0, 0.05)
    my = None
    if np.isfinite(y_lo) and np.isfinite(y_hi) and y_lo <= y_hi:
        my = (ys0 >= y_lo) & (ys0 <= y_hi)
        print(f"{title} [trim y 5%] | points N={int(np.sum(my))} | out={out_trim_y}")
        _scatter_from_arrays(
            xs0[my],
            ys0[my],
            out_trim_y,
            title=f"{title} (trim y 5%)",
            x_label=x_label,
            y_label=y_label,
            loglog=loglog,
            point_size=point_size,
            alpha=alpha,
        )

    # Trim by both size and time (intersection)
    if mx is not None and my is not None:
        mb = mx & my
        print(f"{title} [trim both 5%] | points N={int(np.sum(mb))} | out={out_trim_both}")
        _scatter_from_arrays(
            xs0[mb],
            ys0[mb],
            out_trim_both,
            title=f"{title} (trim x+y 5%)",
            x_label=x_label,
            y_label=y_label,
            loglog=loglog,
            point_size=point_size,
            alpha=alpha,
        )


def _get_x_values(points: List[SolvingPoint], x_axis: str) -> Tuple[np.ndarray, str]:
    if x_axis == "n":
        return np.array([p.num_vars for p in points], dtype=float), "Variable count (n)"
    return np.array([p.size_of_cnf for p in points], dtype=float), "Clause count (m)"


def _get_y_values(points: List[SolvingPoint], y_axis: str) -> Tuple[np.ndarray, str]:
    if y_axis == "m":
        return np.array([p.size_of_cnf for p in points], dtype=float), "Clause count (m)"
    return np.array([p.solving_time for p in points], dtype=float), "Solving time (seconds)"


def _format_k_for_filename(k: float) -> str:
    if np.isfinite(k) and abs(k - round(k)) < 1e-12:
        return str(int(round(k)))
    # Keep it filename-friendly
    s = f"{k}".strip()
    s = s.replace(".", "p").replace("-", "m")
    return s


def _extract_entry_for_k(data: Dict, k_target: float, *, tol: float = 1e-12) -> Dict | None:
    """
    Given a solving-times dict {k_str: {solving_time, size_of_cnf}}, return the entry matching k_target.
    Tries exact string matches first, then numeric match with tolerance.
    """
    if not isinstance(data, dict):
        return None

    # Fast paths: exact key matches that are common in this repo (integers as strings)
    if np.isfinite(k_target) and abs(k_target - round(k_target)) < tol:
        k_int = str(int(round(k_target)))
        v = data.get(k_int)
        if isinstance(v, dict):
            return v

    v = data.get(str(k_target))
    if isinstance(v, dict):
        return v

    # Numeric match fallback
    best_v = None
    for k_str, v0 in data.items():
        if not isinstance(v0, dict):
            continue
        try:
            k_val = float(k_str)
        except Exception:
            continue
        if abs(k_val - k_target) <= tol:
            best_v = v0
            break
    return best_v


def load_points_for_k(k_target: float) -> List[SolvingPoint]:
    points: List[SolvingPoint] = []
    files = list(iter_solving_time_files())
    for instance, path in tqdm(files, desc=f"Loading K={k_target}", unit="file"):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        v = _extract_entry_for_k(data, float(k_target))
        if not isinstance(v, dict):
            continue

        family = infer_family(instance)
        try:
            y = float(v.get("solving_time"))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(y):
            continue
        try:
            size_of_cnf = float(v.get("size_of_cnf"))
        except (TypeError, ValueError):
            size_of_cnf = float("nan")
        try:
            num_vars = float(v.get("nvar"))
        except (TypeError, ValueError):
            num_vars = float("nan")
        if not np.isfinite(num_vars):
            print(f"[warn] missing nvar for instance={instance} k={k_target}")
        if not np.isfinite(size_of_cnf) and not np.isfinite(num_vars):
            continue
        points.append(
            SolvingPoint(
                instance=instance,
                family=family,
                size_of_cnf=size_of_cnf,
                num_vars=num_vars,
                solving_time=y,
            )
        )
    return points


def plot_k_scatter(
    points: List[SolvingPoint],
    out_path: str,
    *,
    k_target: float,
    loglog: bool = False,
    x_axis: str = "m",
    y_axis: str = "time",
) -> None:
    xs, x_label = _get_x_values(points, x_axis)
    ys_raw, y_label = _get_y_values(points, y_axis)
    _plot_scatter_variants(
        xs,
        ys_raw,
        out_path,
        title=f"Point cloud at K={k_target}",
        x_label=x_label,
        y_label=y_label,
        loglog=loglog,
        point_size=14,
        alpha=0.35,
    )


def plot_k_trend_line(
    points: List[SolvingPoint],
    out_path: str,
    *,
    k_target: float,
    bins: int = 30,
    loglog: bool = False,
    x_axis: str = "m",
    y_axis: str = "time",
) -> None:
    # At fixed K: each instance contributes at most one point, so "family lines"
    # (one line per family) is not meaningful. Instead, draw a single trend line
    # using binned medians over all points.
    print(f"Instances with K={k_target}: {len(points)}")
    plt.figure(figsize=(18, 12))
    xs, x_label = _get_x_values(points, x_axis)
    ys, y_label = _get_y_values(points, y_axis)
    x_line, y_line = _family_binned_curve(xs, ys, bins=bins)
    if x_line.size >= 2:
        plt.plot(x_line, y_line, linewidth=3.0, alpha=0.95, color="tab:blue")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Trend line at K={k_target} (binned medians)")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_scatter(
    points: List[SolvingPoint],
    out_path: str,
    *,
    loglog: bool = True,
    x_axis: str = "m",
    y_axis: str = "time",
) -> None:
    xs, x_label = _get_x_values(points, x_axis)
    ys_raw, y_label = _get_y_values(points, y_axis)
    _plot_scatter_variants(
        xs,
        ys_raw,
        out_path,
        title="All prepared points (point cloud)",
        x_label=x_label,
        y_label=y_label,
        loglog=loglog,
        point_size=8,
        alpha=0.18,
    )


def _family_binned_curve(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    bins: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin by log10(x) into 'bins' buckets and return (x_med, y_med) per non-empty bin.
    """
    xs = xs.astype(float)
    ys = _sanitize_positive(ys.astype(float))
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0)
    xs = xs[mask]
    ys = ys[mask]
    if xs.size == 0:
        return np.array([]), np.array([])

    logx = np.log10(xs)
    lo, hi = float(np.min(logx)), float(np.max(logx))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        # Degenerate: fall back to sorting raw points and downsample by unique x
        order = np.argsort(xs)
        xs_s, ys_s = xs[order], ys[order]
        return xs_s, ys_s

    edges = np.linspace(lo, hi, bins + 1)
    x_out: List[float] = []
    y_out: List[float] = []
    for i in range(bins):
        m = (logx >= edges[i]) & (logx < edges[i + 1] if i < bins - 1 else logx <= edges[i + 1])
        if not np.any(m):
            continue
        x_out.append(float(np.nanmedian(xs[m])))
        y_out.append(float(np.nanmedian(ys[m])))

    if not x_out:
        return np.array([]), np.array([])

    x_arr = np.array(x_out)
    y_arr = np.array(y_out)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def plot_family_lines(
    points: List[SolvingPoint],
    out_path: str,
    *,
    bins: int = 30,
    min_family_points: int = 10,
    loglog: bool = True,
    x_axis: str = "m",
    y_axis: str = "time",
) -> None:
    by_family: Dict[str, List[SolvingPoint]] = {}
    for p in points:
        by_family.setdefault(p.family, []).append(p)

    # Filter tiny families (keeps plot readable; user can set min_family_points=1)
    families = [(fam, pts) for fam, pts in by_family.items() if len(pts) >= min_family_points]
    families.sort(key=lambda kv: (-len(kv[1]), kv[0]))
    print(f"Total families (all points): {len(by_family)}")
    print(f"Families plotted (>= {min_family_points} points): {len(families)}")

    # Bigger canvas: many families can be dense
    plt.figure(figsize=(18, 12))
    # More distinct colors even for many lines
    colors = plt.cm.hsv(np.linspace(0, 1, max(1, len(families)), endpoint=False))

    for idx, (fam, pts) in enumerate(families):
        xs, _x_label = _get_x_values(pts, x_axis)
        ys, _y_label = _get_y_values(pts, y_axis)
        x_line, y_line = _family_binned_curve(xs, ys, bins=bins)
        if x_line.size < 2:
            continue
        color = colors[idx]
        # No labels/legend (too cluttered). Emphasize visibility.
        plt.plot(x_line, y_line, linewidth=2.4, alpha=0.9, color=color)

    x_label = _get_x_values(points, x_axis)[1]
    y_label = _get_y_values(points, y_axis)[1]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} scaling by family (binned medians; min points={min_family_points})")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw aggregated solving-time plots")
    parser.add_argument("--log", action="store_true", help="Use log-log axes (linear plots will NOT be generated)")
    parser.add_argument("--also-log", action="store_true", help="Also generate log-log versions (in addition to linear)")
    parser.add_argument("--k", type=float, default=None, help="Only plot points at this K (one point per instance if present)")
    parser.add_argument("--with-lines", action="store_true", help="Also generate line plots (family lines / trend)")
    parser.add_argument("--family-bins", type=int, default=30, help="Bins for family aggregation (log10 size)")
    parser.add_argument(
        "--x",
        type=str,
        default="m",
        choices=["n", "m"],
        help="X axis: n=variable count, m=clause count",
    )
    parser.add_argument(
        "--y",
        type=str,
        default="time",
        choices=["time", "m"],
        help="Y axis: time=solving time, m=clause count",
    )
    parser.add_argument(
        "--min-family-points",
        type=int,
        default=10,
        help="Only draw family lines with at least this many points (set 1 to draw all)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="all_solving_times",
        help="Output file prefix under results/plots/",
    )
    args = parser.parse_args()

    points = load_all_solving_points()
    if not points:
        raise SystemExit("No solving-time points found under results/solving_times/*.json")

    bins = max(5, int(args.family_bins))
    min_family_points = max(1, int(args.min_family_points))
    x_axis = args.x
    y_axis = args.y
    axis_tag = f"x{x_axis}_y{y_axis}"

    # Mode selection:
    # - default: linear only
    # - --log: log only
    # - --also-log: linear + log
    want_linear = not args.log
    want_log = bool(args.log or args.also_log)
    want_lines = bool(args.with_lines)

    if want_linear:
        scatter_path = get_plot_path(f"{args.out_prefix}_{axis_tag}_scatter")
        plot_all_scatter(points, scatter_path, loglog=False, x_axis=x_axis, y_axis=y_axis)
        print(f"Saved scatter plot to: {scatter_path}")
        if want_lines:
            family_path = get_plot_path(f"{args.out_prefix}_{axis_tag}_family_lines")
            plot_family_lines(
                points,
                family_path,
                bins=bins,
                min_family_points=min_family_points,
                loglog=False,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            print(f"Saved family-lines plot to: {family_path}")

    if want_log:
        scatter_log_path = get_plot_path(f"{args.out_prefix}_{axis_tag}_scatter_log")
        plot_all_scatter(points, scatter_log_path, loglog=True, x_axis=x_axis, y_axis=y_axis)
        print(f"Saved scatter (log) plot to: {scatter_log_path}")
        if want_lines:
            family_log_path = get_plot_path(f"{args.out_prefix}_{axis_tag}_family_lines_log")
            plot_family_lines(
                points,
                family_log_path,
                bins=bins,
                min_family_points=min_family_points,
                loglog=True,
                x_axis=x_axis,
                y_axis=y_axis,
            )
            print(f"Saved family-lines (log) plot to: {family_log_path}")

    # Optional: plot only points at a specified K (across all families)
    if args.k is not None:
        k_points = load_points_for_k(float(args.k))
        if not k_points:
            raise SystemExit(f"No points found for K={args.k} in results/solving_times/*.json")

        k_tag = _format_k_for_filename(float(args.k))
        if want_linear:
            out_k = get_plot_path(f"{args.out_prefix}_{axis_tag}_K{k_tag}_scatter")
            plot_k_scatter(k_points, out_k, k_target=float(args.k), loglog=False, x_axis=x_axis, y_axis=y_axis)
            print(f"Saved K-scatter plot to: {out_k}")

            if want_lines:
                out_k_lines = get_plot_path(f"{args.out_prefix}_{axis_tag}_K{k_tag}_trend_line")
                plot_k_trend_line(
                    k_points,
                    out_k_lines,
                    k_target=float(args.k),
                    bins=bins,
                    loglog=False,
                    x_axis=x_axis,
                    y_axis=y_axis,
                )
                print(f"Saved K-trend-line plot to: {out_k_lines}")
        if want_log:
            out_k_log = get_plot_path(f"{args.out_prefix}_{axis_tag}_K{k_tag}_scatter_log")
            plot_k_scatter(k_points, out_k_log, k_target=float(args.k), loglog=True, x_axis=x_axis, y_axis=y_axis)
            print(f"Saved K-scatter (log) plot to: {out_k_log}")

            if want_lines:
                out_k_lines_log = get_plot_path(f"{args.out_prefix}_{axis_tag}_K{k_tag}_trend_line_log")
                plot_k_trend_line(
                    k_points,
                    out_k_lines_log,
                    k_target=float(args.k),
                    bins=bins,
                    loglog=True,
                    x_axis=x_axis,
                    y_axis=y_axis,
                )
                print(f"Saved K-trend-line (log) plot to: {out_k_lines_log}")


if __name__ == "__main__":
    main()
