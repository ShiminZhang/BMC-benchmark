#!/usr/bin/env python3
# primal_cutwidth_approx.py
from __future__ import annotations

import argparse
import random
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


def iter_clauses_dimacs(path: Path) -> Tuple[int, Iterator[List[int]]]:
    """
    Stream DIMACS CNF clauses. Returns (nvars_hint, iterator of clauses as list of var ids).
    - Each clause yielded is a list of variable ids (positive ints), signs removed.
    - Duplicates in a clause are kept initially; caller may dedup.
    """
    nvars_hint = 0

    def gen() -> Iterator[List[int]]:
        nonlocal nvars_hint
        clause: List[int] = []
        with path.open("rb") as f:
            for raw in f:
                if not raw or raw == b"\n" or raw.startswith(b"c"):
                    continue
                if raw.startswith(b"p"):
                    parts = raw.split()
                    # p cnf <nvars> <nclauses>
                    if len(parts) >= 4 and parts[1] == b"cnf":
                        try:
                            nvars_hint = int(parts[2])
                        except Exception:
                            pass
                    continue

                for tok in raw.split():
                    lit = int(tok)
                    if lit == 0:
                        if clause:
                            yield clause
                            clause = []
                    else:
                        clause.append(abs(lit))

        if clause:
            yield clause

    return nvars_hint, gen()

def iter_dimacs_events(path: Path) -> Tuple[int, Iterator[Tuple[str, Union[str, List[int]]]]]:
    """
    Stream a DIMACS CNF file as events:
      - ("iter", <label>) for lines starting with 'c iter ...'
      - ("clause", <vars_list>) for each clause (list of var ids; abs(lit); duplicates preserved)

    Returns (nvars_hint, iterator).
    """
    nvars_hint = 0

    def gen() -> Iterator[Tuple[str, Union[str, List[int]]]]:
        nonlocal nvars_hint
        clause: List[int] = []
        with path.open("rb") as f:
            for raw in f:
                if not raw or raw == b"\n":
                    continue

                if raw.startswith(b"c"):
                    # Only interpret iteration boundaries; other comments ignored.
                    if raw.startswith(b"c iter"):
                        parts = raw.split()
                        if len(parts) >= 3:
                            label = parts[2].decode("utf-8", errors="replace")
                        else:
                            label = str(0)
                        yield ("iter", label)
                    continue

                if raw.startswith(b"p"):
                    parts = raw.split()
                    # p cnf <nvars> <nclauses>
                    if len(parts) >= 4 and parts[1] == b"cnf":
                        try:
                            nvars_hint = int(parts[2])
                        except Exception:
                            pass
                    continue

                if raw.startswith(b"v"):
                    continue

                for tok in raw.split():
                    lit = int(tok)
                    if lit == 0:
                        if clause:
                            yield ("clause", clause)
                            clause = []
                    else:
                        clause.append(abs(lit))

        if clause:
            yield ("clause", clause)

    return nvars_hint, gen()


def build_pos_map(n: int, order: str, seed: int, trial: int) -> List[int]:
    """
    pos[var] = position in [1..n]
    """
    if order == "natural":
        pos = [0] * (n + 1)
        for v in range(1, n + 1):
            pos[v] = v
        return pos

    # random
    rng = random.Random(seed + trial * 1000003)
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    pos = [0] * (n + 1)
    for i, v in enumerate(perm, start=1):
        pos[v] = i
    return pos


def primal_cutwidth_under_pos(
    cnf: Path,
    n: int,
    pos: List[int],
    clause_cap: Optional[int] = None,
    skip_large: bool = False,
) -> int:
    """
    Compute cutwidth of the primal graph under the given variable ordering (pos[]),
    WITHOUT explicitly building the primal graph edges.

    For a clause with sorted positions p1 < ... < pk:
      For cuts i in [pj, p(j+1)-1], #left=j, #right=k-j, contribution = j*(k-j).
    Sum contributions over clauses; take max over i=1..n-1.

    Memory: O(n) for diff array.
    """
    diff = [0] * (n + 2)  # diff[i] applies to cut i; we scan 1..n-1

    # stream clauses
    _, clauses = iter_clauses_dimacs(cnf)
    for vars_list in clauses:
        if not vars_list:
            continue

        # optional cap/skip for huge clauses (engineering approximation)
        if clause_cap is not None and len(vars_list) > clause_cap:
            if skip_large:
                continue
            # else: truncate (still an approximation)
            vars_list = vars_list[:clause_cap]

        # map to positions; dedup vars within clause (important)
        # using set is OK per clause; clause sizes are usually moderate
        s = set()
        for v in vars_list:
            if 1 <= v <= n:
                s.add(pos[v])
        if len(s) <= 1:
            continue

        ps = sorted(s)
        k = len(ps)
        # add piecewise-constant contributions
        for j in range(1, k):  # j = #vars on left side for cuts between ps[j-1] and ps[j]
            l = ps[j - 1]
            r = ps[j] - 1
            if l > n - 1:
                break
            if r < 1:
                continue
            if r > n - 1:
                r = n - 1
            if l < 1:
                l = 1
            if l <= r:
                val = j * (k - j)
                diff[l] += val
                diff[r + 1] -= val

    # prefix to find max
    best = 0
    cur = 0
    for i in range(1, n):
        cur += diff[i]
        if cur > best:
            best = cur
    return best

def primal_cutwidth_complete_under_pos(
    cnf: Path,
    n: int,
    pos: List[int],
) -> int:
    """
    Exact cutwidth of the (unweighted) primal graph under the given ordering (pos[]).

    Primal graph: an (undirected) edge {u,v} exists iff u and v appear together in any clause.
    For each edge with positions p<q, it contributes +1 to all cuts i in [p, q-1].
    """
    if n <= 1:
        return 0

    diff = [0] * (n + 2)
    seen_edges: set[int] = set()  # encode (p<<32)|q where p<q in positions

    _, clauses = iter_clauses_dimacs(cnf)
    for vars_list in clauses:
        if not vars_list:
            continue

        ps_set = set()
        for v in vars_list:
            if 1 <= v <= n:
                ps_set.add(pos[v])
        if len(ps_set) <= 1:
            continue

        ps = sorted(ps_set)
        m = len(ps)
        for i in range(m - 1):
            p = ps[i]
            for j in range(i + 1, m):
                q = ps[j]
                key = (p << 32) | q
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                if p < q:
                    diff[p] += 1
                    diff[q] -= 1  # cuts are 1..n-1; q corresponds to (q-1)+1 boundary in diff form

    best = 0
    cur = 0
    for i in range(1, n):
        cur += diff[i]
        if cur > best:
            best = cur
    return best

def _cutwidth_from_edge_positions(n: int, pos_edges: Iterator[Tuple[int, int]]) -> int:
    """
    Given edges expressed in *positions* (p,q) with p and q in [1..n], compute cutwidth:
    max over cuts i=1..n-1 of number of edges with p<=i<q (assuming p<q).
    """
    if n <= 1:
        return 0
    diff = [0] * (n + 2)
    for p, q in pos_edges:
        if p == q:
            continue
        if p > q:
            p, q = q, p
        if p < 1 or q < 1 or p > n or q > n:
            continue
        diff[p] += 1
        diff[q] -= 1
    best = 0
    cur = 0
    for i in range(1, n):
        cur += diff[i]
        if cur > best:
            best = cur
    return best

def _nx_import() -> Any:
    try:
        import networkx as nx  # type: ignore[import-not-found]
        return nx
    except Exception as e:
        raise SystemExit(
            "NetworkX backend requested, but 'networkx' is not available. "
            "Install it (e.g., pip install networkx) or use --backend stream."
        ) from e

def _cutwidth_via_networkx_cut_size(g: Any, n: int, pos: List[int]) -> int:
    """
    Compute cutwidth under ordering pos[] using NetworkX's built-in cut function:
    max_{i=1..n-1} cut_size(G, S_i) where S_i = {v | pos[v] <= i}.

    Notes:
    - This validates against NetworkX's cut accounting (`nx.cut_size`), not a custom edge-scan.
    - Nodes absent from g do not affect cut sizes (no incident edges).
    """
    nx = _nx_import()
    if n <= 1:
        return 0
    nodes_by_pos: List[int] = [0] * (n + 1)
    for v in range(1, n + 1):
        p = pos[v]
        if 1 <= p <= n:
            nodes_by_pos[p] = v

    left: set[int] = set()
    best = 0
    # scan cuts between i and i+1 => i in [1..n-1]
    for i in range(1, n):
        v = nodes_by_pos[i]
        if v in g:
            left.add(v)
        # T=None => complement of left in g
        cw = int(nx.cut_size(g, left))
        if cw > best:
            best = cw
    return best

def _block_size_stats(cnf: Path, n: int) -> List[Dict[str, Any]]:
    """
    Per 'c iter' block, collect size metrics that are useful for scaling analysis:
    - clauses: number of clauses in the block
    - lits: total literals in the block (counting duplicates as they appear in CNF)
    - vars: number of distinct variables appearing in the block
    """
    _, events = iter_dimacs_events(cnf)
    out: List[Dict[str, Any]] = []

    cur_label = "0"
    block_idx = 0
    cur_has_clause = False
    clause_cnt = 0
    lit_cnt = 0
    var_set: set[int] = set()

    def flush() -> None:
        nonlocal clause_cnt, lit_cnt, var_set, cur_has_clause, block_idx
        if not cur_has_clause:
            return
        out.append(
            {
                "block_idx": block_idx,
                "iter": cur_label,
                "clauses": clause_cnt,
                "lits": lit_cnt,
                "vars": len(var_set),
            }
        )
        block_idx += 1
        clause_cnt = 0
        lit_cnt = 0
        var_set = set()
        cur_has_clause = False

    for kind, payload in events:
        if kind == "iter":
            flush()
            cur_label = str(payload)
            continue
        # clause
        cur_has_clause = True
        vars_list = payload if isinstance(payload, list) else []
        clause_cnt += 1
        lit_cnt += len(vars_list)
        for v in vars_list:
            if 1 <= v <= n:
                var_set.add(v)

    flush()
    return out

def _block_edge_list_networkx(cnf: Path, n: int) -> List[int]:
    """
    Per block (in order), count primal-graph edges using NetworkX (deduplicated).
    """
    graphs = _build_primal_graph_blocks_nx(cnf, n)
    return [int(g.number_of_edges()) for _, g in graphs]

def _load_summary_instances(summary_json: Path) -> Tuple[Path, Path, List[str], int, int]:
    """
    Load a summary JSON produced by this script's --all-instances mode.
    Returns (aig_dir, cnf_dir, instance_names, k_min, k_max).
    """
    import json

    payload = json.loads(summary_json.read_text())
    aig_dir = Path(payload.get("aig_dir", "data/aigs"))
    cnf_dir = Path(payload.get("cnf_dir", "data/cnfs"))
    k_min = int(payload.get("k_min", 0))
    k_max = int(payload.get("k_max", 0))
    rows = payload.get("rows", [])
    instances: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("status") != "ok":
            continue
        inst = r.get("instance")
        if isinstance(inst, str) and inst:
            instances.append(inst)
    return aig_dir, cnf_dir, instances, k_min, k_max

def _maybe_plot_scaling_scatter(
    *,
    rows: List[Dict[str, Any]],
    out_png: Path,
    title: str,
    x_key: str,
    loglog: bool = True,
) -> bool:
    """
    rows: list of dicts with at least keys x_key and 'cutwidth'
    Produces a scatter plot (all points). Returns True on success.
    """
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return False

    xs: List[float] = []
    ys: List[float] = []
    for r in rows:
        try:
            x = float(r.get(x_key, 0))
            y = float(r.get("cutwidth", 0))
        except Exception:
            continue
        if loglog:
            if x <= 0 or y <= 0:
                continue
        xs.append(x)
        ys.append(y)

    if not xs or not ys:
        return False

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    # All points; black solid dots for visibility.
    plt.scatter(xs, ys, s=6, alpha=1.0, linewidths=0, c="black")
    plt.xlabel(x_key)
    plt.ylabel("cutwidth")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    if loglog:
        plt.xscale("log")
        plt.yscale("log")

    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()
    return True

def _build_primal_adj_lists(cnf: Path, n: int) -> List[List[int]]:
    """
    Build primal graph adjacency lists (undirected, deduplicated) as Python lists.
    Memory can be large for big cliques (large clauses), same as exact primal graph.
    """
    adj: List[set[int]] = [set() for _ in range(n + 1)]
    _, clauses = iter_clauses_dimacs(cnf)
    for vars_list in clauses:
        if not vars_list:
            continue
        vs = set()
        for v in vars_list:
            if 1 <= v <= n:
                vs.add(v)
        if len(vs) <= 1:
            continue
        for u, v in combinations(vs, 2):
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
    return [sorted(list(s)) for s in adj]

def _pos_from_order(order: List[int], n: int) -> List[int]:
    pos = [0] * (n + 1)
    for i, v in enumerate(order, start=1):
        if 1 <= v <= n:
            pos[v] = i
    return pos

def _cutwidth_from_adj_under_pos(adj: List[List[int]], n: int, pos: List[int]) -> int:
    def gen_pos_edges() -> Iterator[Tuple[int, int]]:
        for u in range(1, n + 1):
            pu = pos[u]
            if pu <= 0:
                continue
            for v in adj[u]:
                if v > u:
                    pv = pos[v]
                    if pv > 0:
                        yield (pu, pv)

    return _cutwidth_from_edge_positions(n, gen_pos_edges())

def _ordering_bfs_by_degree(adj: List[List[int]], n: int) -> List[int]:
    from collections import deque

    deg = [0] * (n + 1)
    for v in range(1, n + 1):
        deg[v] = len(adj[v])
    unvisited = [True] * (n + 1)
    order: List[int] = []
    q: "deque[int]" = deque()

    def pick_start() -> Optional[int]:
        best_v = None
        best_d = None
        for v in range(1, n + 1):
            if not unvisited[v]:
                continue
            d = deg[v]
            if best_v is None or d < (best_d if best_d is not None else d):
                best_v = v
                best_d = d
        return best_v

    while len(order) < n:
        s = pick_start()
        if s is None:
            break
        unvisited[s] = False
        q.append(s)
        while q:
            u = q.popleft()
            order.append(u)
            nbrs = [v for v in adj[u] if unvisited[v]]
            nbrs.sort(key=lambda x: (deg[x], x))
            for v in nbrs:
                unvisited[v] = False
                q.append(v)
    return order

def _ordering_dfs_by_degree(adj: List[List[int]], n: int) -> List[int]:
    deg = [0] * (n + 1)
    for v in range(1, n + 1):
        deg[v] = len(adj[v])
    unvisited = [True] * (n + 1)
    order: List[int] = []

    def pick_start() -> Optional[int]:
        best_v = None
        best_d = None
        for v in range(1, n + 1):
            if not unvisited[v]:
                continue
            d = deg[v]
            if best_v is None or d < (best_d if best_d is not None else d):
                best_v = v
                best_d = d
        return best_v

    while len(order) < n:
        s = pick_start()
        if s is None:
            break
        stack: List[int] = [s]
        unvisited[s] = False
        while stack:
            u = stack.pop()
            order.append(u)
            nbrs = [v for v in adj[u] if unvisited[v]]
            nbrs.sort(key=lambda x: (deg[x], x), reverse=True)  # stack LIFO => push higher first
            for v in nbrs:
                unvisited[v] = False
                stack.append(v)
    return order

def _ordering_min_degree_elimination(adj: List[List[int]], n: int) -> List[int]:
    import heapq

    active = [True] * (n + 1)
    deg = [0] * (n + 1)
    for v in range(1, n + 1):
        deg[v] = len(adj[v])
    heap: List[Tuple[int, int]] = [(deg[v], v) for v in range(1, n + 1)]
    heapq.heapify(heap)
    order: List[int] = []
    while heap and len(order) < n:
        d, v = heapq.heappop(heap)
        if not active[v]:
            continue
        if d != deg[v]:
            continue
        active[v] = False
        order.append(v)
        for u in adj[v]:
            if active[u]:
                deg[u] -= 1
                heapq.heappush(heap, (deg[u], u))
    # if something went odd, append any missing
    if len(order) < n:
        for v in range(1, n + 1):
            if v not in set(order):
                order.append(v)
    return order

def _ordering_max_degree_elimination(adj: List[List[int]], n: int) -> List[int]:
    import heapq

    active = [True] * (n + 1)
    deg = [0] * (n + 1)
    for v in range(1, n + 1):
        deg[v] = len(adj[v])
    heap: List[Tuple[int, int]] = [(-deg[v], v) for v in range(1, n + 1)]
    heapq.heapify(heap)
    order: List[int] = []
    while heap and len(order) < n:
        nd, v = heapq.heappop(heap)
        if not active[v]:
            continue
        d = -nd
        if d != deg[v]:
            continue
        active[v] = False
        order.append(v)
        for u in adj[v]:
            if active[u]:
                deg[u] -= 1
                heapq.heappush(heap, (-deg[u], u))
    if len(order) < n:
        seen = set(order)
        for v in range(1, n + 1):
            if v not in seen:
                order.append(v)
    return order

def _ordering_random(n: int, seed: int, trial: int) -> List[int]:
    rng = random.Random(seed + trial * 1000003)
    perm = list(range(1, n + 1))
    rng.shuffle(perm)
    return perm

def _best_ordering_auto(
    cnf: Path,
    *,
    n: int,
    seed: int,
    trials: int,
    backend: str,
) -> Tuple[str, List[int], int]:
    """
    Try several heuristics and return (algo_name, best_order, best_cutwidth).
    Requires exact primal graph notion (edges), so this is meant for --complete.
    """
    # User requested: do NOT use custom orderings; only NetworkX orderings.
    if backend != "networkx":
        raise SystemExit("--order auto requires --backend networkx")
    _ = seed, trials  # reserved for future NetworkX orderings that use randomness

    _nx_import()
    g = _build_primal_graph_nx(cnf, n)

    candidates: List[Tuple[str, List[int]]] = []
    # NOTE: per user request, auto currently tests ONLY reverse Cuthill–McKee.
    # Keep other NetworkX orderings commented-out for easy re-enable later.
    #
    # # Spectral ordering (may fail if linear algebra backend is unavailable).
    # try:
    #     from networkx.linalg.algebraicconnectivity import spectral_ordering  # type: ignore
    #     candidates.append(("spectral", list(spectral_ordering(g))))
    # except Exception:
    #     pass
    #
    # # Cuthill-McKee (non-reversed)
    # try:
    #     from networkx.utils import cuthill_mckee_ordering  # type: ignore
    #     candidates.append(("cm", list(cuthill_mckee_ordering(g))))
    # except Exception:
    #     pass
    #
    # Reverse Cuthill-McKee
    try:
        from networkx.utils import reverse_cuthill_mckee_ordering  # type: ignore

        candidates.append(("rcm", list(reverse_cuthill_mckee_ordering(g))))
    except Exception:
        pass

    # Fallback if all NetworkX orderings failed.
    if not candidates:
        candidates.append(("nodelist_sorted", sorted(g.nodes())))

    best_name = candidates[0][0]
    best_order = candidates[0][1]
    best_val: Optional[int] = None

    for name, order in candidates:
        pos = _pos_from_order(order, n)
        val = _cutwidth_via_networkx_cut_size(g, n, pos)
        if best_val is None or val < best_val:
            best_val = val
            best_name = name
            best_order = order

    return best_name, best_order, int(best_val) if best_val is not None else 0

def _best_ordering_auto_block(
    cnf: Path,
    *,
    n: int,
    seed: int,
    trials: int,
    backend: str,
) -> Tuple[str, List[int], List[Tuple[str, int]], int]:
    """
    Try several heuristics and return (algo_name, best_order, best_blocks, best_score),
    where best_score = max(block cutwidth) (the same objective used elsewhere for block mode).
    """
    # User requested: do NOT use custom orderings; only NetworkX orderings.
    if backend != "networkx":
        raise SystemExit("--order auto requires --backend networkx")
    _ = seed, trials  # reserved for future NetworkX orderings that use randomness

    _nx_import()
    g_all = _build_primal_graph_nx(cnf, n)
    nx_block_graphs: List[Tuple[str, Any]] = _build_primal_graph_blocks_nx(cnf, n)

    candidates: List[Tuple[str, List[int]]] = []
    # NOTE: per user request, auto currently tests ONLY reverse Cuthill–McKee.
    # Keep other NetworkX orderings commented-out for easy re-enable later.
    #
    # try:
    #     from networkx.linalg.algebraicconnectivity import spectral_ordering  # type: ignore
    #     candidates.append(("spectral", list(spectral_ordering(g_all))))
    # except Exception:
    #     pass
    # try:
    #     from networkx.utils import cuthill_mckee_ordering  # type: ignore
    #     candidates.append(("cm", list(cuthill_mckee_ordering(g_all))))
    # except Exception:
    #     pass
    try:
        from networkx.utils import reverse_cuthill_mckee_ordering  # type: ignore

        candidates.append(("rcm", list(reverse_cuthill_mckee_ordering(g_all))))
    except Exception:
        pass
    if not candidates:
        candidates.append(("nodelist_sorted", sorted(g_all.nodes())))

    best_name = candidates[0][0]
    best_order = candidates[0][1]
    best_blocks: List[Tuple[str, int]] = []
    best_score: Optional[int] = None

    for name, order in candidates:
        pos = _pos_from_order(order, n)
        blocks = [(lab, _cutwidth_via_networkx_cut_size(g, n, pos)) for lab, g in nx_block_graphs]

        if not blocks:
            # No iter markers -> fallback to whole formula score
            score = _cutwidth_via_networkx_cut_size(g_all, n, pos)
            blocks = [("all", int(score))]
        score2 = max((v for _, v in blocks), default=0)

        if best_score is None or score2 < best_score:
            best_score = score2
            best_name = name
            best_order = order
            best_blocks = [(lab, int(v)) for lab, v in blocks]

    return best_name, best_order, best_blocks, int(best_score) if best_score is not None else 0

def _build_primal_graph_nx(cnf: Path, n: int) -> Any:
    """
    Build the primal graph using NetworkX (deduplicated, unweighted undirected graph).
    """
    nx = _nx_import()
    g = nx.Graph()
    # Include isolated variables as nodes for ordering routines.
    g.add_nodes_from(range(1, n + 1))
    _, clauses = iter_clauses_dimacs(cnf)
    for vars_list in clauses:
        if not vars_list:
            continue
        vs = set()
        for v in vars_list:
            if 1 <= v <= n:
                vs.add(v)
        if len(vs) <= 1:
            continue
        # Add clique edges induced by this clause.
        g.add_edges_from(combinations(vs, 2))
    return g

def _build_primal_graph_blocks_nx(cnf: Path, n: int) -> List[Tuple[str, Any]]:
    """
    Build per-block primal graphs separated by 'c iter <label>' markers, using NetworkX.
    """
    nx = _nx_import()
    _, events = iter_dimacs_events(cnf)
    out: List[Tuple[str, Any]] = []

    cur_label = "0"
    cur_g = nx.Graph()
    cur_g.add_nodes_from(range(1, n + 1))
    cur_has_clause = False

    for kind, payload in events:
        if kind == "iter":
            if cur_has_clause:
                out.append((cur_label, cur_g))
            cur_label = str(payload)
            cur_g = nx.Graph()
            cur_g.add_nodes_from(range(1, n + 1))
            cur_has_clause = False
            continue

        # clause
        cur_has_clause = True
        vars_list = payload if isinstance(payload, list) else []
        vs = set()
        for v in vars_list:
            if 1 <= v <= n:
                vs.add(v)
        if len(vs) <= 1:
            continue
        cur_g.add_edges_from(combinations(vs, 2))

    if cur_has_clause:
        out.append((cur_label, cur_g))
    return out

def primal_cutwidth_complete_networkx_under_pos(cnf: Path, n: int, pos: List[int]) -> int:
    """
    Exact primal-graph cutwidth under the given ordering, using NetworkX:
    - Build primal graph as `nx.Graph()` (deduplicated undirected edges)
    - Compute cutwidth via NetworkX's built-in `nx.cut_size()` over all prefix cuts
    """
    g = _build_primal_graph_nx(cnf, n)
    return _cutwidth_via_networkx_cut_size(g, n, pos)

def primal_cutwidth_complete_networkx_blocks_under_pos(
    cnf: Path, n: int, pos: List[int]
) -> List[Tuple[str, int]]:
    """
    Exact per-block primal-graph cutwidth under the given ordering, using NetworkX:
    graphs are split by 'c iter' markers; cutwidth computed via `nx.cut_size()`.
    """
    graphs = _build_primal_graph_blocks_nx(cnf, n)
    return [
        (lab, _cutwidth_via_networkx_cut_size(g, n, pos))
        for lab, g in graphs
    ]

def primal_cutwidth_complete_blocks_under_pos(
    cnf: Path,
    n: int,
    pos: List[int],
) -> List[Tuple[str, int]]:
    """
    Exact cutwidth per block separated by comment lines 'c iter <label> ...',
    using the primal graph definition with edge de-duplication.
    """
    _, events = iter_dimacs_events(cnf)
    out: List[Tuple[str, int]] = []

    cur_label = "0"
    cur_diff: Dict[int, int] = {}
    cur_seen: set[int] = set()
    cur_has_clause = False

    for kind, payload in events:
        if kind == "iter":
            if cur_has_clause:
                out.append((cur_label, _max_from_sparse_diff(cur_diff, n)))
            cur_label = str(payload)
            cur_diff = {}
            cur_seen = set()
            cur_has_clause = False
            continue

        # clause
        cur_has_clause = True
        vars_list = payload if isinstance(payload, list) else []
        ps_set = set()
        for v in vars_list:
            if 1 <= v <= n:
                ps_set.add(pos[v])
        if len(ps_set) <= 1:
            continue
        ps = sorted(ps_set)
        m = len(ps)
        for i in range(m - 1):
            p = ps[i]
            for j in range(i + 1, m):
                q = ps[j]
                key = (p << 32) | q
                if key in cur_seen:
                    continue
                cur_seen.add(key)
                # edge crosses cuts in [p, q-1] => sparse diff add at p and (q)
                cur_diff[p] = cur_diff.get(p, 0) + 1
                cur_diff[q] = cur_diff.get(q, 0) - 1

    if cur_has_clause:
        out.append((cur_label, _max_from_sparse_diff(cur_diff, n)))
    return out


def infer_n_if_needed(cnf: Path, n_hint: int) -> int:
    """
    If header nvars is missing/bad, infer max variable id by streaming once.
    """
    if n_hint > 0:
        return n_hint
    mx = 0
    _, clauses = iter_clauses_dimacs(cnf)
    for vars_list in clauses:
        for v in vars_list:
            if v > mx:
                mx = v
    return mx

def _add_clause_contribs_to_sparse_diff(
    diff: Dict[int, int],
    vars_list: List[int],
    n: int,
    pos: List[int],
    clause_cap: Optional[int],
    skip_large: bool,
) -> None:
    if not vars_list:
        return

    if clause_cap is not None and len(vars_list) > clause_cap:
        if skip_large:
            return
        vars_list = vars_list[:clause_cap]

    s = set()
    for v in vars_list:
        if 1 <= v <= n:
            s.add(pos[v])
    if len(s) <= 1:
        return

    ps = sorted(s)
    k = len(ps)
    for j in range(1, k):
        l = ps[j - 1]
        r = ps[j] - 1
        if l > n - 1:
            break
        if r < 1:
            continue
        if r > n - 1:
            r = n - 1
        if l < 1:
            l = 1
        if l <= r:
            val = j * (k - j)
            diff[l] = diff.get(l, 0) + val
            diff[r + 1] = diff.get(r + 1, 0) - val

def _max_from_sparse_diff(diff: Dict[int, int], n: int) -> int:
    """
    Given a sparse diff-map (difference array updates), compute max prefix value
    over cuts i=1..n-1.
    """
    if n <= 1 or not diff:
        return 0
    keys = [k for k in diff.keys() if 1 <= k <= n - 1]
    if not keys:
        return 0
    keys.sort()
    cur = 0
    best = 0
    for k in keys:
        cur += diff[k]
        if cur > best:
            best = cur
    return best

def primal_cutwidth_blocks_under_pos(
    cnf: Path,
    n: int,
    pos: List[int],
    clause_cap: Optional[int] = None,
    skip_large: bool = False,
) -> List[Tuple[str, int]]:
    """
    Compute cutwidth per block separated by comment lines 'c iter <label> ...'.

    Each block is the set of clauses between consecutive 'c iter' markers. Clauses
    before the first marker (if any) are treated as label '0'.
    """
    n_hint, events = iter_dimacs_events(cnf)
    _ = n_hint  # n already provided/inferred by caller

    out: List[Tuple[str, int]] = []
    cur_label = "0"
    cur_diff: Dict[int, int] = {}
    cur_has_clause = False

    for kind, payload in events:
        if kind == "iter":
            # boundary: finalize previous block if it had any clauses
            if cur_has_clause:
                out.append((cur_label, _max_from_sparse_diff(cur_diff, n)))
            cur_label = str(payload)
            cur_diff = {}
            cur_has_clause = False
            continue

        # clause
        cur_has_clause = True
        _add_clause_contribs_to_sparse_diff(
            cur_diff,
            payload if isinstance(payload, list) else [],
            n,
            pos,
            clause_cap,
            skip_large,
        )

    if cur_has_clause:
        out.append((cur_label, _max_from_sparse_diff(cur_diff, n)))
    return out

def _compute_single(
    cnf_path: Path,
    *,
    n: int,
    order: str,
    trials: int,
    seed: int,
    mode: str,
    complete: bool,
    backend: str,
    clause_cap: Optional[int],
    skip_large: bool,
) -> Union[int, List[Tuple[str, int]]]:
    """
    Compute cutwidth for one CNF.
    - mode == "all": returns int
    - mode == "block": returns list[(label, value)]
    """
    if order == "auto":
        if not complete:
            raise SystemExit("--order auto requires --complete (cutwidth is defined on the primal graph edges)")
        if mode == "all":
            _, _, best_val = _best_ordering_auto(
                cnf_path, n=n, seed=seed, trials=trials, backend=backend
            )
            return int(best_val)
        # block mode
        _, _, best_blocks, _best_score = _best_ordering_auto_block(
            cnf_path, n=n, seed=seed, trials=trials, backend=backend
        )
        return best_blocks

    if order == "natural":
        pos = build_pos_map(n, "natural", seed, 0)
        if complete:
            if backend == "networkx":
                if mode == "block":
                    return primal_cutwidth_complete_networkx_blocks_under_pos(cnf_path, n, pos)
                return primal_cutwidth_complete_networkx_under_pos(cnf_path, n, pos)
            if mode == "block":
                return primal_cutwidth_complete_blocks_under_pos(cnf_path, n, pos)
            return primal_cutwidth_complete_under_pos(cnf_path, n, pos)

        if mode == "block":
            return primal_cutwidth_blocks_under_pos(
                cnf_path, n, pos, clause_cap=clause_cap, skip_large=skip_large
            )
        return primal_cutwidth_under_pos(
            cnf_path, n, pos, clause_cap=clause_cap, skip_large=skip_large
        )

    # random: use best (minimum) value as approximation of optimal cutwidth
    if mode == "all":
        best_val: Optional[int] = None
        nx_graph = None
        if complete and backend == "networkx":
            nx_graph = _build_primal_graph_nx(cnf_path, n)
        for t in range(trials):
            pos = build_pos_map(n, "random", seed, t)
            if complete:
                if backend == "networkx":
                    assert nx_graph is not None
                    val = _cutwidth_from_edge_positions(
                        n, ((pos[u], pos[v]) for u, v in nx_graph.edges())
                    )
                else:
                    val = primal_cutwidth_complete_under_pos(cnf_path, n, pos)
            else:
                val = primal_cutwidth_under_pos(
                    cnf_path, n, pos, clause_cap=clause_cap, skip_large=skip_large
                )
            if best_val is None or val < best_val:
                best_val = val
        return best_val if best_val is not None else 0

    # random + block: choose trial that minimizes max(blocks)
    best_blocks: Optional[List[Tuple[str, int]]] = None
    best_score: Optional[int] = None
    nx_block_graphs: Optional[List[Tuple[str, Any]]] = None
    if complete and backend == "networkx":
        nx_block_graphs = _build_primal_graph_blocks_nx(cnf_path, n)
    for t in range(trials):
        pos = build_pos_map(n, "random", seed, t)
        if complete:
            if backend == "networkx":
                assert nx_block_graphs is not None
                blocks = [
                    (
                        lab,
                        _cutwidth_from_edge_positions(
                            n, ((pos[u], pos[v]) for u, v in g.edges())
                        ),
                    )
                    for lab, g in nx_block_graphs
                ]
            else:
                blocks = primal_cutwidth_complete_blocks_under_pos(cnf_path, n, pos)
        else:
            blocks = primal_cutwidth_blocks_under_pos(
                cnf_path, n, pos, clause_cap=clause_cap, skip_large=skip_large
            )
        if not blocks:
            # fallback: treat as all-mode
            if complete:
                if backend == "networkx":
                    # No blocks: compute on whole-graph.
                    g_all = _build_primal_graph_nx(cnf_path, n)
                    score = _cutwidth_from_edge_positions(
                        n, ((pos[u], pos[v]) for u, v in g_all.edges())
                    )
                else:
                    score = primal_cutwidth_complete_under_pos(cnf_path, n, pos)
            else:
                score = primal_cutwidth_under_pos(
                    cnf_path, n, pos, clause_cap=clause_cap, skip_large=skip_large
                )
            blocks = [("all", score)]
        else:
            score = max(v for _, v in blocks) if blocks else 0
        if best_score is None or score < best_score:
            best_score = score
            best_blocks = blocks
    return best_blocks if best_blocks is not None else []

def _maybe_plot_series(
    xs: List[int],
    series: Dict[str, List[Optional[float]]],
    out_png: Path,
    title: str,
    y_label: str = "cutwidth",
) -> bool:
    try:
        import matplotlib  # type: ignore[import-not-found]
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
        from matplotlib.ticker import MaxNLocator  # type: ignore[import-not-found]
    except Exception:
        plt = None  # type: ignore[assignment]

    out_png.parent.mkdir(parents=True, exist_ok=True)

    # Preferred: matplotlib (nicer); fallback: PIL (always available on many clusters).
    if plt is not None:
        plt.figure(figsize=(7.5, 4.3))
        # Plot each series; skip missing (None) points
        for name, ys in series.items():
            x_ok: List[int] = []
            y_ok: List[float] = []
            for x, y in zip(xs, ys):
                if y is None:
                    continue
                x_ok.append(x)
                y_ok.append(float(y))
            if not x_ok:
                continue
            plt.plot(x_ok, y_ok, marker="o", linewidth=1.2, markersize=3, alpha=0.9)
        ax = plt.gca()
        ax.set_xlabel("k")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        # x-axis as integers only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return True

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False

    if not xs or not series:
        return False

    # gather all y values
    all_y: List[float] = []
    for ys in series.values():
        for y in ys:
            if y is not None:
                all_y.append(float(y))
    if not all_y:
        return False

    W, H = 900, 520
    left, top, right, bottom = 70, 40, 30, 70
    pw = W - left - right
    ph = H - top - bottom

    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # ranges
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = float(min(all_y)), float(max(all_y))
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1

    def xpix(x: int) -> int:
        return int(left + (x - xmin) * pw / (xmax - xmin))

    def ypix(y: float) -> int:
        # y increases downwards in image
        return int(top + (ymax - y) * ph / (ymax - ymin))

    # axes
    ax0 = (left, top)
    ax1 = (left, top + ph)
    ax2 = (left + pw, top + ph)
    draw.line([ax1, ax2], fill="black", width=2)
    draw.line([ax1, ax0], fill="black", width=2)

    # ticks
    xticks = sorted(set(xs))
    # reduce tick count if too many
    if len(xticks) > 12:
        step = max(1, len(xticks) // 10)
        xticks = xticks[::step]
        if xticks[-1] != xs[-1]:
            xticks.append(xs[-1])

    for x in xticks:
        xp = xpix(x)
        draw.line([(xp, top + ph), (xp, top + ph + 6)], fill="black", width=1)
        draw.text((xp - 6, top + ph + 10), str(x), fill="black", font=font)

    yticks = 5
    for t in range(yticks + 1):
        yv = ymin + (ymax - ymin) * t / yticks
        yp = ypix(float(yv))
        draw.line([(left - 6, yp), (left, yp)], fill="black", width=1)
        draw.text((5, yp - 6), str(int(yv)), fill="black", font=font)

    # labels/title
    draw.text((left, 10), title, fill="black", font=font)
    draw.text((W // 2 - 10, H - 30), "k", fill="black", font=font)
    draw.text((10, 10), y_label, fill="black", font=font)

    palette = [
        (30, 90, 200),
        (220, 50, 32),
        (35, 160, 95),
        (150, 90, 200),
        (240, 160, 40),
        (0, 140, 170),
        (120, 120, 120),
    ]

    # polylines
    for si, (name, ys) in enumerate(series.items()):
        color = palette[si % len(palette)]
        pts: List[Tuple[int, int]] = []
        for x, y in zip(xs, ys):
            if y is None:
                continue
            pts.append((xpix(x), ypix(float(y))))
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                draw.line([pts[i - 1], pts[i]], fill=color, width=2)
        for (xp, yp) in pts:
            r = 2
            draw.ellipse([(xp - r, yp - r), (xp + r, yp + r)], outline="black", fill=color)

    img.save(out_png)
    return True

def _maybe_plot(xs: List[int], ys: List[int], out_png: Path, title: str) -> bool:
    series = {"cutwidth": [float(y) for y in ys]}
    return _maybe_plot_series(xs, series, out_png, title, y_label="cutwidth")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf", type=Path, nargs="?", default=None,
                    help="DIMACS CNF file (optional if --instance is provided)")
    ap.add_argument("--instance", type=str, default=None,
                    help="Instance name (e.g., 139442p0). If set, will generate CNF via simplecar then compute cutwidth.")
    ap.add_argument("--all-instances", action="store_true",
                    help="Run for all instances found under --aig-dir (all *.aig). "
                         "With --k-max: run k-range per instance (JSON+plots + summary). "
                         "With --k: run a fixed k and emit a first-iteration table + plot.")
    ap.add_argument("--limit-instances", type=int, default=0,
                    help="If >0, only run the first N instances (useful for quick smoke tests).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Alias for --limit-instances.")
    ap.add_argument("--k", type=int, default=None,
                    help="BMC bound k (required with --instance). CNF assumed at <cnf-dir>/<instance>.<k>.cnf")
    ap.add_argument("--k-max", type=int, default=None,
                    help="If set with --instance, run for all k in [k-min..k-max], generate CNFs, compute cutwidths, save results and plot.")
    ap.add_argument("--k-min", type=int, default=2,
                    help="Start k for --k-max range (default: 2)")
    ap.add_argument("--k-step", type=int, default=1,
                    help="Step for --k-max range (default: 1)")
    ap.add_argument("--aig-dir", type=Path, default=Path("./data/aigs"),
                    help="Directory containing <instance>.aig (used with --instance)")
    ap.add_argument("--cnf-dir", type=Path, default=Path("./data/cnfs"),
                    help="Directory to place/read generated CNFs (used with --instance)")
    ap.add_argument("--simplecar", type=Path, default=Path("./libs/bin/simplecar"),
                    help="Path to simplecar binary (used with --instance)")
    ap.add_argument("--simplecar-verbose", action="store_true",
                    help="Show simplecar stdout/stderr (otherwise suppressed).")
    ap.add_argument("--force-generate", action="store_true",
                    help="Force re-generate CNF even if it already exists (used with --instance)")
    ap.add_argument("--mode", choices=["all", "block"], default="all",
                    help="all: one cutwidth for whole formula; block: per 'c iter' block (TSV output)")
    ap.add_argument("--block-stats", action="store_true",
                    help="When --mode block, include per-block size stats (clauses/vars/lits, and edges if using --backend networkx). "
                         "Output becomes TSV with header.")
    ap.add_argument("--from-summary", type=Path, default=None,
                    help="Read a summary JSON (from --all-instances) and run for all instances at a fixed k. "
                         "Requires --mode block. Outputs an aggregated TSV (see --out-tsv).")
    ap.add_argument("--from-tsv", type=Path, default=None,
                    help="Plot directly from an existing TSV (requires --scaling-x). "
                         "TSV must include a 'cutwidth' column and the chosen x column.")
    ap.add_argument("--summary-k", type=int, default=0,
                    help="k to use with --from-summary. If 0, will use summary's k_min when k_min==k_max.")
    ap.add_argument("--out-tsv", type=Path, default=None,
                    help="Output TSV path for --from-summary (default under ./results/cutwidth_scaling/).")
    ap.add_argument("--scaling-x", choices=["clauses", "vars", "lits", "edges"], default="clauses",
                    help="When using --from-summary, x-axis for scaling plot (default: clauses).")
    ap.add_argument("--scaling-scope", choices=["first_block", "all_blocks"], default="first_block",
                    help="When using --from-summary: first_block=one point per instance (block_idx=0); "
                         "all_blocks=all blocks of all instances (many points). Default: first_block.")
    ap.add_argument("--no-scaling-plot", action="store_true",
                    help="Disable scaling plot generation for --from-summary.")
    ap.add_argument("--logscale", action="store_true",
                    help="Enable log-log scaling for scaling plots (default: off).")
    ap.add_argument("--no-logscale", action="store_true",
                    help="Disable log-log scaling for scaling plots (deprecated; default is off).")
    ap.add_argument("--no-generate-missing-cnfs", action="store_true",
                    help="When using --from-summary, do not generate missing CNFs via simplecar; just skip them.")
    ap.add_argument("--block-plot", choices=["profile", "blocks", "avg", "min", "max"], default="profile",
                    help="When --mode block and running k-range, how to plot: "
                         "profile=K lines, one per k (x=block index 0..k); "
                         "blocks=one line per block index across k; "
                         "avg=single line of average block cutwidth; "
                         "min=single line of min block cutwidth; max=single line of max block cutwidth.")
    ap.add_argument("--complete", action="store_true",
                    help="Compute exact primal-graph cutwidth for the given ordering (de-duplicate edges). Slower than default.")
    ap.add_argument("--backend", choices=["stream", "networkx"], default="stream",
                    help="cutwidth computation backend: stream=existing implementation; "
                         "networkx=build primal graph with NetworkX then compute cutwidth (requires --complete).")
    ap.add_argument("--order", choices=["natural", "random", "auto"], default="natural",
                    help="natural: use variable id order; random: sample random orderings; "
                         "auto: try NetworkX orderings (spectral/cm/rcm) and pick the best (requires --backend networkx).")
    ap.add_argument("--trials", type=int, default=1,
                    help="number of random trials (only used if --order random)")
    ap.add_argument("--seed", type=int, default=1, help="random seed base")
    ap.add_argument("--nvars", type=int, default=0,
                    help="override number of variables (otherwise from header or inferred)")
    ap.add_argument("--clause-cap", type=int, default=None,
                    help="cap clause length (approx). If set, either truncate or skip.")
    ap.add_argument("--skip-large", action="store_true",
                    help="if set with --clause-cap, skip clauses larger than cap (instead of truncating)")
    ap.add_argument("--out-json", type=Path, default=None,
                    help="Output JSON path for --k-max run (default under ./results/cutwidth/)")
    ap.add_argument("--out-plot", type=Path, default=None,
                    help="Output plot PNG path for --k-max run (default under ./results/plots/)")
    ap.add_argument("--summary-json", type=Path, default=None,
                    help="Summary JSON path for --all-instances (default under ./results/cutwidth/)")
    args = ap.parse_args()
    # Normalize limit flag.
    if args.limit and not args.limit_instances:
        args.limit_instances = args.limit

    def iter_with_tqdm(items: List[str], desc: str) -> Iterator[str]:
        try:
            from tqdm import tqdm  # type: ignore[import-not-found]

            return iter(tqdm(items, desc=desc))
        except Exception:
            return iter(items)

    # Plot-only mode from existing TSV.
    if args.from_tsv is not None:
        import csv

        tsv_path = args.from_tsv
        if not tsv_path.exists():
            raise SystemExit(f"TSV not found: {tsv_path}")
        rows: List[Dict[str, Any]] = []
        with tsv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            headers = reader.fieldnames or []
            if "cutwidth" not in headers:
                raise SystemExit(f"TSV missing required column: cutwidth (found: {headers})")
            if args.scaling_x not in headers:
                hint = ""
                if "clauses" in headers and args.scaling_x == "lits":
                    hint = " (hint: use --scaling-x clauses)"
                raise SystemExit(
                    f"TSV missing required column: {args.scaling_x} (found: {headers}){hint}"
                )
            for row in reader:
                rows.append(row)
        if not rows:
            raise SystemExit(f"TSV has no data rows: {tsv_path}")
        # auto output path
        out_png = tsv_path.with_suffix(".png")
        ok_plot = _maybe_plot_scaling_scatter(
            rows=rows,
            out_png=out_png,
            title=f"{tsv_path.stem} scaling ({args.scaling_x})",
            x_key=args.scaling_x,
            loglog=(args.logscale and not args.no_logscale),
        )
        if ok_plot:
            print(f"plot\t{out_png}")
        else:
            print("plot\tskipped(no matplotlib)")
        return

    if args.complete and (args.clause_cap is not None or args.skip_large):
        ap.error("--complete cannot be used with --clause-cap/--skip-large (would break exactness).")
    if args.backend == "networkx" and not args.complete:
        ap.error("--backend networkx requires --complete (NetworkX builds a deduplicated primal graph).")
    if args.order == "auto" and not args.complete:
        ap.error("--order auto requires --complete")
    if args.order == "auto" and args.backend != "networkx":
        ap.error("--order auto requires --backend networkx (no custom orderings)")

    # Bulk mode: read instances from summary and compute per-iteration cutwidth + size stats.
    if args.from_summary is not None:
        if args.mode != "block":
            ap.error("--from-summary requires --mode block")
        sum_aig_dir, sum_cnf_dir, instances, k_min, k_max = _load_summary_instances(args.from_summary)
        if args.limit_instances and args.limit_instances > 0:
            instances = instances[: args.limit_instances]
        k = int(args.summary_k)
        if k <= 0:
            if k_min > 0 and k_min == k_max:
                k = k_min
            else:
                ap.error("--summary-k is required when summary has a k-range")

        # Prefer CLI dirs (if user overrides), otherwise use summary dirs.
        aig_dir = args.aig_dir if args.aig_dir is not None else sum_aig_dir
        cnf_dir = args.cnf_dir if args.cnf_dir is not None else sum_cnf_dir

        out_dir = Path("./results/cutwidth_scaling")
        default_out = out_dir / f"{args.from_summary.stem}.k{k}.{args.mode}.{args.order}.{args.backend}.{args.scaling_x}.{args.scaling_scope}.tsv"
        out_tsv = args.out_tsv if args.out_tsv is not None else default_out
        out_tsv.parent.mkdir(parents=True, exist_ok=True)

        # Header: flat table
        # - first_block: one row per instance
        # - all_blocks: one row per instance x block
        lines: List[str] = ["instance\tk\tblock_idx\titer\tcutwidth\tclauses\tvars\tlits\tedges"]
        plot_rows: List[Dict[str, Any]] = []

        generated = 0
        skipped_missing = 0
        processed = 0

        for inst in iter_with_tqdm(instances, desc="instances"):
            cnf_path = cnf_dir / f"{inst}.{k}.cnf"
            if not cnf_path.exists():
                # Try alternative common naming: <inst>.k<k>.cnf
                alt = cnf_dir / f"{inst}.k{k}.cnf"
                cnf_path = alt if alt.exists() else cnf_path
            if not cnf_path.exists():
                if args.no_generate_missing_cnfs:
                    skipped_missing += 1
                    continue
                # Attempt to generate CNF from AIG using simplecar.
                aig_path = aig_dir / f"{inst}.aig"
                if not aig_path.exists():
                    skipped_missing += 1
                    continue
                cnf_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(args.simplecar),
                    "-bmc",
                    "-k",
                    str(k),
                    "-cnf",
                    str(cnf_dir) + "/",
                    str(aig_path),
                ]
                if args.simplecar_verbose:
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                generated += 1
                # simplecar's naming matches <inst>.<k>.cnf
                cnf_path = cnf_dir / f"{inst}.{k}.cnf"
                if not cnf_path.exists():
                    # fallback: skip if generation didn't produce expected path
                    skipped_missing += 1
                    continue

            n_hint, _ = iter_clauses_dimacs(cnf_path)
            n = args.nvars if args.nvars > 0 else infer_n_if_needed(cnf_path, n_hint)
            if n <= 1:
                continue

            # Compute blocks with requested ordering/backend.
            val_or_blocks = _compute_single(
                cnf_path,
                n=n,
                order=args.order,
                trials=args.trials,
                seed=args.seed,
                mode="block",
                complete=args.complete,
                backend=args.backend,
                clause_cap=args.clause_cap,
                skip_large=args.skip_large,
            )
            blocks = val_or_blocks if isinstance(val_or_blocks, list) else [("all", int(val_or_blocks))]

            stats_list = _block_size_stats(cnf_path, n)
            edges_list: List[int] = []
            if args.complete and args.backend == "networkx":
                edges_list = _block_edge_list_networkx(cnf_path, n)

            if not blocks:
                continue
            processed += 1

            if args.scaling_scope == "first_block":
                i, (lab, cw) = 0, blocks[0]
                s = stats_list[0] if len(stats_list) > 0 else {}
                edges = edges_list[0] if len(edges_list) > 0 else 0
                row = {
                    "instance": inst,
                    "k": k,
                    "block_idx": 0,
                    "iter": lab,
                    "cutwidth": int(cw),
                    "clauses": int(s.get("clauses", 0)),
                    "vars": int(s.get("vars", 0)),
                    "lits": int(s.get("lits", 0)),
                    "edges": int(edges),
                }
                plot_rows.append(row)
                lines.append(
                    f"{inst}\t{k}\t0\t{lab}\t{int(cw)}\t"
                    f"{int(s.get('clauses', 0))}\t{int(s.get('vars', 0))}\t{int(s.get('lits', 0))}\t{int(edges)}"
                )
            else:
                for i, (lab, cw) in enumerate(blocks):
                    s = stats_list[i] if i < len(stats_list) else {}
                    edges = edges_list[i] if i < len(edges_list) else 0
                    row = {
                        "instance": inst,
                        "k": k,
                        "block_idx": i,
                        "iter": lab,
                        "cutwidth": int(cw),
                        "clauses": int(s.get("clauses", 0)),
                        "vars": int(s.get("vars", 0)),
                        "lits": int(s.get("lits", 0)),
                        "edges": int(edges),
                    }
                    plot_rows.append(row)
                    lines.append(
                        f"{inst}\t{k}\t{i}\t{lab}\t{int(cw)}\t"
                        f"{int(s.get('clauses', 0))}\t{int(s.get('vars', 0))}\t{int(s.get('lits', 0))}\t{int(edges)}"
                    )

        out_tsv.write_text("\n".join(lines) + "\n")
        print(f"saved\t{out_tsv}")

        if not args.no_scaling_plot:
            out_png = out_tsv.with_suffix(".png")
            ok_plot = _maybe_plot_scaling_scatter(
                rows=plot_rows,
                out_png=out_png,
                title=f"{args.from_summary.stem} k={k} {args.scaling_scope} scaling ({args.order}, {args.backend})",
                x_key=args.scaling_x,
                loglog=(args.logscale and not args.no_logscale),
            )
            if ok_plot:
                print(f"plot\t{out_png}")
            else:
                print("plot\tskipped(no matplotlib)")
        return

    if args.all_instances:
        if args.instance is not None or args.cnf is not None:
            ap.error("--all-instances cannot be combined with positional cnf/--instance")
        if args.k is not None and args.k_max is not None:
            ap.error("--all-instances: use either --k (fixed) or --k-max (range), not both")
        if args.k is None and args.k_max is None:
            ap.error("--all-instances requires --k (fixed) or --k-max (range)")

        # Fixed-k mode: generate CNF for each instance at k, compute first-iteration cutwidth + size.
        if args.k is not None:
            if args.mode != "block":
                ap.error("--all-instances with --k requires --mode block (first iteration)")

            k = int(args.k)
            if k < 1:
                ap.error("--k must be >= 1")

            # discover instances from AIG dir
            aigs = sorted(args.aig_dir.glob("*.aig"))
            instances = [p.stem for p in aigs]
            if args.limit_instances and args.limit_instances > 0:
                instances = instances[:args.limit_instances]
            if not instances:
                raise SystemExit(f"No .aig files found under: {args.aig_dir}")

            out_dir = Path("./results/cutwidth_scaling")
            default_out = out_dir / f"ALL.k{k}.first_block.{args.mode}.{args.order}.{args.backend}.tsv"
            out_tsv = args.out_tsv if args.out_tsv is not None else default_out
            out_tsv.parent.mkdir(parents=True, exist_ok=True)

            lines: List[str] = ["instance\tk\titer\tcutwidth\tclauses"]
            plot_rows: List[Dict[str, Any]] = []

            generated = 0
            skipped_missing = 0
            processed = 0

            for inst in instances:
                cnf_path = args.cnf_dir / f"{inst}.{k}.cnf"
                if not cnf_path.exists():
                    alt = args.cnf_dir / f"{inst}.k{k}.cnf"
                    cnf_path = alt if alt.exists() else cnf_path
                if not cnf_path.exists():
                    # Attempt to generate CNF from AIG using simplecar.
                    aig_path = args.aig_dir / f"{inst}.aig"
                    if not aig_path.exists():
                        skipped_missing += 1
                        continue
                    args.cnf_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        str(args.simplecar),
                        "-bmc",
                        "-k",
                        str(k),
                        "-cnf",
                        str(args.cnf_dir) + "/",
                        str(aig_path),
                    ]
                    if args.simplecar_verbose:
                        subprocess.run(cmd, check=True)
                    else:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    generated += 1
                    cnf_path = args.cnf_dir / f"{inst}.{k}.cnf"
                    if not cnf_path.exists():
                        skipped_missing += 1
                        continue

                n_hint, _ = iter_clauses_dimacs(cnf_path)
                n = args.nvars if args.nvars > 0 else infer_n_if_needed(cnf_path, n_hint)
                if n <= 1:
                    continue

                # Compute block cutwidths and take the first block.
                val_or_blocks = _compute_single(
                    cnf_path,
                    n=n,
                    order=args.order,
                    trials=args.trials,
                    seed=args.seed,
                    mode="block",
                    complete=args.complete,
                    backend=args.backend,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )
                blocks = val_or_blocks if isinstance(val_or_blocks, list) else [("all", int(val_or_blocks))]
                if not blocks:
                    continue
                lab, cw = blocks[0]

                stats_list = _block_size_stats(cnf_path, n)
                s0 = stats_list[0] if len(stats_list) > 0 else {}
                clauses = int(s0.get("clauses", 0))

                plot_rows.append(
                    {
                        "instance": inst,
                        "k": k,
                        "iter": lab,
                        "cutwidth": int(cw),
                        "clauses": clauses,
                    }
                )
                lines.append(f"{inst}\t{k}\t{lab}\t{int(cw)}\t{clauses}")
                processed += 1

            out_tsv.write_text("\n".join(lines) + "\n")
            print(f"saved\t{out_tsv}")
            print(f"instances\t{processed}\tgenerated\t{generated}\tskipped_missing\t{skipped_missing}")

            if not args.no_scaling_plot:
                out_png = out_tsv.with_suffix(".png")
                ok_plot = _maybe_plot_scaling_scatter(
                    rows=plot_rows,
                    out_png=out_png,
                    title=f"ALL k={k} first_block ({args.order}, {args.backend})",
                    x_key="clauses",
                    loglog=(args.logscale and not args.no_logscale),
                )
                if ok_plot:
                    print(f"plot\t{out_png}")
                else:
                    print("plot\tskipped(no matplotlib)")
            return


        # discover instances from AIG dir
        aigs = sorted(args.aig_dir.glob("*.aig"))
        instances = [p.stem for p in aigs]
        if args.limit_instances and args.limit_instances > 0:
            instances = instances[:args.limit_instances]
        if not instances:
            raise SystemExit(f"No .aig files found under: {args.aig_dir}")

        results_dir = Path("./results/cutwidth")
        default_summary = results_dir / f"ALL.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.complete{int(args.complete)}.summary.json"
        summary_path = args.summary_json if args.summary_json is not None else default_summary

        summary_rows: List[Dict[str, Any]] = []
        ok = 0
        fail = 0
        for inst in iter_with_tqdm(instances, desc="instances"):
            try:
                # Reuse the same logic as range mode by emulating --instance.
                # Determine output paths (per instance).
                plots_dir = Path("./results/plots")
                out_json = results_dir / f"{inst}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.json"
                if args.complete:
                    out_json = results_dir / f"{inst}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.complete.json"
                if args.complete and args.backend == "networkx":
                    out_json = out_json.with_name(out_json.stem + ".networkx" + out_json.suffix)
                out_png = plots_dir / f"{inst}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.png"
                if args.complete:
                    out_png = plots_dir / f"{inst}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.complete.png"
                if args.complete and args.backend == "networkx":
                    out_png = out_png.with_name(out_png.stem + ".networkx" + out_png.suffix)
                if args.mode == "block" and args.block_plot != "profile":
                    out_png = out_png.with_name(out_png.stem + f".{args.block_plot}" + out_png.suffix)

                # Run ks for this instance (mostly copied from range-mode branch)
                aig_path = (args.aig_dir / f"{inst}.aig")
                if not aig_path.exists():
                    raise FileNotFoundError(str(aig_path))

                rows: List[Dict[str, Any]] = []
                xs: List[int] = []
                ys: List[int] = []

                for k in range(args.k_min, args.k_max + 1, args.k_step):
                    out_cnf = (args.cnf_dir / f"{inst}.{k}.cnf")
                    if args.force_generate or not out_cnf.exists():
                        args.cnf_dir.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            str(args.simplecar),
                            "-bmc",
                            "-k",
                            str(k),
                            "-cnf",
                            str(args.cnf_dir) + "/",
                            str(aig_path),
                        ]
                        if args.simplecar_verbose:
                            subprocess.run(cmd, check=True)
                        else:
                            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    if not out_cnf.exists():
                        raise FileNotFoundError(str(out_cnf))

                    n_hint, _ = iter_clauses_dimacs(out_cnf)
                    n = args.nvars if args.nvars > 0 else infer_n_if_needed(out_cnf, n_hint)
                    if n <= 1:
                        val_or_blocks: Union[int, List[Tuple[str, int]]] = 0 if args.mode == "all" else [("0", 0)]
                    else:
                        val_or_blocks = _compute_single(
                            out_cnf,
                            n=n,
                            order=args.order,
                            trials=args.trials,
                            seed=args.seed,
                            mode=args.mode,
                            complete=args.complete,
                            backend=args.backend,
                            clause_cap=args.clause_cap,
                            skip_large=args.skip_large,
                        )

                    if args.mode == "all":
                        v = int(val_or_blocks)  # type: ignore[arg-type]
                        rows.append({"k": k, "cutwidth": v})
                        xs.append(k)
                        ys.append(v)
                    else:
                        blocks = val_or_blocks if isinstance(val_or_blocks, list) else [("all", int(val_or_blocks))]
                        plot_v = max((b for _, b in blocks), default=0)
                        avg_v = (sum(v for _, v in blocks) / len(blocks)) if blocks else 0.0
                        min_v = min((v for _, v in blocks), default=0)
                        stats_list: List[Dict[str, Any]] = []
                        edges_list: List[int] = []
                        if args.block_stats:
                            stats_list = _block_size_stats(out_cnf, n)
                            if args.complete and args.backend == "networkx":
                                edges_list = _block_edge_list_networkx(out_cnf, n)
                        rows.append({
                            "k": k,
                            "plot_cutwidth": plot_v,
                            "avg_block_cutwidth": avg_v,
                            "min_block_cutwidth": min_v,
                            "blocks": [
                                {
                                    "block_idx": i,
                                    "iter": lab,
                                    "cutwidth": v,
                                    **(
                                        {
                                            "clauses": int((stats_list[i] if i < len(stats_list) else {}).get("clauses", 0)),
                                            "vars": int((stats_list[i] if i < len(stats_list) else {}).get("vars", 0)),
                                            "lits": int((stats_list[i] if i < len(stats_list) else {}).get("lits", 0)),
                                            "edges": int(edges_list[i] if i < len(edges_list) else 0),
                                        }
                                        if args.block_stats
                                        else {}
                                    ),
                                }
                                for i, (lab, v) in enumerate(blocks)
                            ],
                        })
                        xs.append(k)
                        ys.append(plot_v)

                out_json.parent.mkdir(parents=True, exist_ok=True)
                import json
                payload: Dict[str, Any] = {
                    "instance": inst,
                    "aig": str(aig_path),
                    "cnf_dir": str(args.cnf_dir),
                    "mode": args.mode,
                    "complete": bool(args.complete),
                    "order": args.order,
                    "trials": args.trials,
                    "seed": args.seed,
                    "k_min": args.k_min,
                    "k_max": args.k_max,
                    "k_step": args.k_step,
                    "rows": rows,
                }
                out_json.write_text(json.dumps(payload, indent=2))
                if args.mode == "block":
                    if args.block_plot == "avg":
                        avg_series = {"avg": [float(r.get("avg_block_cutwidth", 0.0)) for r in rows]}
                        plotted = _maybe_plot_series(
                            xs,
                            avg_series,
                            out_png,
                            title=f"{inst} avg block cutwidth ({args.order}{', complete' if args.complete else ''})",
                            y_label="avg block cutwidth",
                        )
                    elif args.block_plot == "min":
                        min_series = {"min": [float(r.get("min_block_cutwidth", 0)) for r in rows]}
                        plotted = _maybe_plot_series(
                            xs,
                            min_series,
                            out_png,
                            title=f"{inst} min block cutwidth ({args.order}{', complete' if args.complete else ''})",
                            y_label="min block cutwidth",
                        )
                    elif args.block_plot == "max":
                        max_series = {"max": [float(r.get("plot_cutwidth", 0)) for r in rows]}
                        plotted = _maybe_plot_series(
                            xs,
                            max_series,
                            out_png,
                            title=f"{inst} max block cutwidth ({args.order}{', complete' if args.complete else ''})",
                            y_label="max block cutwidth",
                        )
                    elif args.block_plot == "blocks":
                        # Plot one line per block index (block 0..B-1), x-axis is k.
                        max_blocks = 0
                        for r in rows:
                            max_blocks = max(max_blocks, len(r.get("blocks", [])))
                        series: Dict[str, List[Optional[float]]] = {
                            f"b{i}": [None] * len(xs) for i in range(max_blocks)
                        }
                        for t, r in enumerate(rows):
                            blks = r.get("blocks", [])
                            for i, b in enumerate(blks):
                                if i < max_blocks:
                                    series[f"b{i}"][t] = float(b["cutwidth"])
                        plotted = _maybe_plot_series(
                            xs,
                            series,
                            out_png,
                            title=f"{inst} block cutwidths ({args.order}{', complete' if args.complete else ''})",
                            y_label="block cutwidth",
                        )
                    else:
                        # profile: K lines, one per k; x-axis is block index
                        max_blocks = 0
                        for r in rows:
                            max_blocks = max(max_blocks, len(r.get("blocks", [])))
                        x_block = list(range(max_blocks))
                        series: Dict[str, List[Optional[float]]] = {}
                        for r in rows:
                            k = r.get("k")
                            blks = r.get("blocks", [])
                            ys_prof: List[Optional[float]] = [None] * max_blocks
                            for i, b in enumerate(blks):
                                if i < max_blocks:
                                    ys_prof[i] = float(b["cutwidth"])
                            series[f"k={k}"] = ys_prof
                        plotted = _maybe_plot_series(
                            x_block,
                            series,
                            out_png,
                            title=f"{inst} block profile by k ({args.order}{', complete' if args.complete else ''})",
                            y_label="block cutwidth",
                        )
                else:
                    plotted = _maybe_plot(xs, ys, out_png, title=f"{inst} cutwidth ({args.mode}, {args.order}{', complete' if args.complete else ''})")

                summary_rows.append({
                    "instance": inst,
                    "status": "ok",
                    "json": str(out_json),
                    "plot": str(out_png) if plotted else None,
                })
                ok += 1
                print(f"ok\t{inst}\t{out_json}")
            except Exception as e:
                summary_rows.append({
                    "instance": inst,
                    "status": "fail",
                    "error": f"{type(e).__name__}: {e}",
                })
                fail += 1
                print(f"fail\t{inst}\t{type(e).__name__}: {e}")

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        summary_payload = {
            "aig_dir": str(args.aig_dir),
            "cnf_dir": str(args.cnf_dir),
            "mode": args.mode,
            "complete": bool(args.complete),
            "order": args.order,
            "trials": args.trials,
            "seed": args.seed,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "k_step": args.k_step,
            "ok": ok,
            "fail": fail,
            "rows": summary_rows,
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        print(f"summary\t{summary_path}")
        return

    # Range mode: generate + compute for k in [k-min..k-max]
    if args.instance is not None and args.k_max is not None:
        if args.k_min < 1:
            ap.error("--k-min must be >= 1")
        if args.k_step < 1:
            ap.error("--k-step must be >= 1")
        aig_path = (args.aig_dir / f"{args.instance}.aig")
        if not aig_path.exists():
            raise SystemExit(f"AIG not found: {aig_path}")

        # determine output paths
        results_dir = Path("./results/cutwidth")
        plots_dir = Path("./results/plots")
        default_json = results_dir / f"{args.instance}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.json"
        default_png = plots_dir / f"{args.instance}.k{args.k_min}-{args.k_max}.{args.mode}.{args.order}.png"
        if args.complete and args.backend == "networkx":
            default_json = default_json.with_name(default_json.stem + ".networkx" + default_json.suffix)
            default_png = default_png.with_name(default_png.stem + ".networkx" + default_png.suffix)
        if args.mode == "block" and args.block_plot != "profile":
            default_png = default_png.with_name(default_png.stem + f".{args.block_plot}" + default_png.suffix)
        out_json = args.out_json if args.out_json is not None else default_json
        out_png = args.out_plot if args.out_plot is not None else default_png

        # run ks
        rows: List[Dict[str, Any]] = []
        xs: List[int] = []
        ys: List[int] = []

        for k in range(args.k_min, args.k_max + 1, args.k_step):
            out_cnf = (args.cnf_dir / f"{args.instance}.{k}.cnf")
            if args.force_generate or not out_cnf.exists():
                args.cnf_dir.mkdir(parents=True, exist_ok=True)
                cmd = [
                    str(args.simplecar),
                    "-bmc",
                    "-k",
                    str(k),
                    "-cnf",
                    str(args.cnf_dir) + "/",
                    str(aig_path),
                ]
                if args.simplecar_verbose:
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if not out_cnf.exists():
                raise SystemExit(f"Expected CNF not found after generation: {out_cnf}")

            n_hint, _ = iter_clauses_dimacs(out_cnf)
            n = args.nvars if args.nvars > 0 else infer_n_if_needed(out_cnf, n_hint)
            if n <= 1:
                val_or_blocks: Union[int, List[Tuple[str, int]]] = 0 if args.mode == "all" else [("0", 0)]
            else:
                val_or_blocks = _compute_single(
                    out_cnf,
                    n=n,
                    order=args.order,
                    trials=args.trials,
                    seed=args.seed,
                    mode=args.mode,
                    complete=args.complete,
                    backend=args.backend,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )

            if args.mode == "all":
                v = int(val_or_blocks)  # type: ignore[arg-type]
                rows.append({"k": k, "cutwidth": v})
                xs.append(k)
                ys.append(v)
            else:
                blocks = val_or_blocks if isinstance(val_or_blocks, list) else [("all", int(val_or_blocks))]
                plot_v = max((b for _, b in blocks), default=0)
                avg_v = (sum(v for _, v in blocks) / len(blocks)) if blocks else 0.0
                min_v = min((v for _, v in blocks), default=0)
                stats_list: List[Dict[str, Any]] = []
                edges_list: List[int] = []
                if args.block_stats:
                    stats_list = _block_size_stats(out_cnf, n)
                    if args.complete and args.backend == "networkx":
                        edges_list = _block_edge_list_networkx(out_cnf, n)
                rows.append({
                    "k": k,
                    "plot_cutwidth": plot_v,
                    "avg_block_cutwidth": avg_v,
                    "min_block_cutwidth": min_v,
                    "blocks": [
                        {
                            "block_idx": i,
                            "iter": lab,
                            "cutwidth": v,
                            **(
                                {
                                    "clauses": int((stats_list[i] if i < len(stats_list) else {}).get("clauses", 0)),
                                    "vars": int((stats_list[i] if i < len(stats_list) else {}).get("vars", 0)),
                                    "lits": int((stats_list[i] if i < len(stats_list) else {}).get("lits", 0)),
                                    "edges": int(edges_list[i] if i < len(edges_list) else 0),
                                }
                                if args.block_stats
                                else {}
                            ),
                        }
                        for i, (lab, v) in enumerate(blocks)
                    ],
                })
                xs.append(k)
                ys.append(plot_v)

        # save json
        out_json.parent.mkdir(parents=True, exist_ok=True)
        import json
        payload: Dict[str, Any] = {
            "instance": args.instance,
            "aig": str(aig_path),
            "cnf_dir": str(args.cnf_dir),
            "mode": args.mode,
            "complete": bool(args.complete),
            "order": args.order,
            "trials": args.trials,
            "seed": args.seed,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "k_step": args.k_step,
            "rows": rows,
        }
        out_json.write_text(json.dumps(payload, indent=2))

        # plot
        if args.mode == "block":
            if args.block_plot == "avg":
                avg_series = {"avg": [float(r.get("avg_block_cutwidth", 0.0)) for r in rows]}
                plotted = _maybe_plot_series(
                    xs,
                    avg_series,
                    out_png,
                    title=f"{args.instance} avg block cutwidth ({args.order}{', complete' if args.complete else ''})",
                    y_label="avg block cutwidth",
                )
            elif args.block_plot == "min":
                min_series = {"min": [float(r.get("min_block_cutwidth", 0)) for r in rows]}
                plotted = _maybe_plot_series(
                    xs,
                    min_series,
                    out_png,
                    title=f"{args.instance} min block cutwidth ({args.order}{', complete' if args.complete else ''})",
                    y_label="min block cutwidth",
                )
            elif args.block_plot == "max":
                max_series = {"max": [float(r.get("plot_cutwidth", 0)) for r in rows]}
                plotted = _maybe_plot_series(
                    xs,
                    max_series,
                    out_png,
                    title=f"{args.instance} max block cutwidth ({args.order}{', complete' if args.complete else ''})",
                    y_label="max block cutwidth",
                )
            elif args.block_plot == "blocks":
                max_blocks = 0
                for r in rows:
                    max_blocks = max(max_blocks, len(r.get("blocks", [])))
                series: Dict[str, List[Optional[float]]] = {
                    f"b{i}": [None] * len(xs) for i in range(max_blocks)
                }
                for t, r in enumerate(rows):
                    blks = r.get("blocks", [])
                    for i, b in enumerate(blks):
                        if i < max_blocks:
                            series[f"b{i}"][t] = float(b["cutwidth"])
                plotted = _maybe_plot_series(
                    xs,
                    series,
                    out_png,
                    title=f"{args.instance} block cutwidths ({args.order}{', complete' if args.complete else ''})",
                    y_label="block cutwidth",
                )
            else:
                # profile: K lines, one per k; x-axis is block index
                max_blocks = 0
                for r in rows:
                    max_blocks = max(max_blocks, len(r.get("blocks", [])))
                x_block = list(range(max_blocks))
                series: Dict[str, List[Optional[float]]] = {}
                for r in rows:
                    k = r.get("k")
                    blks = r.get("blocks", [])
                    ys_prof: List[Optional[float]] = [None] * max_blocks
                    for i, b in enumerate(blks):
                        if i < max_blocks:
                            ys_prof[i] = float(b["cutwidth"])
                    series[f"k={k}"] = ys_prof
                plotted = _maybe_plot_series(
                    x_block,
                    series,
                    out_png,
                    title=f"{args.instance} block profile by k ({args.order}{', complete' if args.complete else ''})",
                    y_label="block cutwidth",
                )
        else:
            plotted = _maybe_plot(xs, ys, out_png, title=f"{args.instance} cutwidth ({args.mode}, {args.order})")
        if plotted:
            print(f"saved\t{out_json}\nplot\t{out_png}")
        else:
            print(f"saved\t{out_json}\nplot\tskipped(no matplotlib)")
        return

    # Resolve CNF path: either positional cnf, or auto-generate from --instance/--k
    cnf_path: Optional[Path] = args.cnf
    if args.instance is not None:
        if args.k is None:
            ap.error("--k is required when using --instance")
        aig_path = (args.aig_dir / f"{args.instance}.aig")
        out_cnf = (args.cnf_dir / f"{args.instance}.{args.k}.cnf")
        cnf_path = out_cnf

        if args.force_generate or not out_cnf.exists():
            # Ensure output directory exists
            args.cnf_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                str(args.simplecar),
                "-bmc",
                "-k",
                str(args.k),
                "-cnf",
                str(args.cnf_dir) + "/",
                str(aig_path),
            ]
            try:
                if args.simplecar_verbose:
                    subprocess.run(cmd, check=True)
                else:
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError as e:
                raise SystemExit(f"Failed to run simplecar: {e}") from e
            except subprocess.CalledProcessError as e:
                raise SystemExit(f"simplecar failed with exit code {e.returncode}") from e

        if not out_cnf.exists():
            raise SystemExit(f"Expected CNF not found after generation: {out_cnf}")

    if cnf_path is None:
        ap.error("Please provide a CNF path, or use --instance with --k to generate one.")

    # get n
    n_hint, _ = iter_clauses_dimacs(cnf_path)
    n = args.nvars if args.nvars > 0 else infer_n_if_needed(cnf_path, n_hint)
    if n <= 1:
        if args.mode == "block":
            # nothing meaningful to split; keep simple
            print("0\t0")
        else:
            print(0)
        return

    if args.order == "natural":
        pos = build_pos_map(n, "natural", args.seed, 0)
        if args.mode == "block":
            if args.complete:
                blocks = (
                    primal_cutwidth_complete_networkx_blocks_under_pos(cnf_path, n, pos)
                    if args.backend == "networkx"
                    else primal_cutwidth_complete_blocks_under_pos(cnf_path, n, pos)
                )
            else:
                blocks = primal_cutwidth_blocks_under_pos(
                    cnf_path, n, pos,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )
            if not blocks:
                # no iter markers detected -> fallback to whole formula
                print(
                    (
                        primal_cutwidth_complete_networkx_under_pos(cnf_path, n, pos)
                        if (args.complete and args.backend == "networkx")
                        else primal_cutwidth_complete_under_pos(cnf_path, n, pos)
                    )
                    if args.complete
                    else primal_cutwidth_under_pos(
                        cnf_path, n, pos,
                        clause_cap=args.clause_cap,
                        skip_large=args.skip_large,
                    )
                )
            else:
                if args.block_stats:
                    stats_list = _block_size_stats(cnf_path, n)
                    edges_list: List[int] = []
                    if args.complete and args.backend == "networkx":
                        edges_list = _block_edge_list_networkx(cnf_path, n)
                    print("block_idx\titer\tcutwidth\tclauses\tvars\tlits\tedges")
                    for i, (lab, v) in enumerate(blocks):
                        s = stats_list[i] if i < len(stats_list) else {}
                        edges = edges_list[i] if i < len(edges_list) else 0
                        print(
                            f"{i}\t{lab}\t{v}\t"
                            f"{int(s.get('clauses', 0))}\t{int(s.get('vars', 0))}\t{int(s.get('lits', 0))}\t{int(edges)}"
                        )
                else:
                    for lab, v in blocks:
                        print(f"{lab}\t{v}")
        else:
            if args.complete:
                val = (
                    primal_cutwidth_complete_networkx_under_pos(cnf_path, n, pos)
                    if args.backend == "networkx"
                    else primal_cutwidth_complete_under_pos(cnf_path, n, pos)
                )
            else:
                val = primal_cutwidth_under_pos(
                    cnf_path, n, pos,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )
            print(val)
        return

    if args.order == "auto":
        # One-shot mode: output the best result among candidate orderings.
        if args.mode == "all":
            _, _, best_val = _best_ordering_auto(
                cnf_path, n=n, seed=args.seed, trials=args.trials, backend=args.backend
            )
            print(best_val)
        else:
            _, _, blocks, _ = _best_ordering_auto_block(
                cnf_path, n=n, seed=args.seed, trials=args.trials, backend=args.backend
            )
            if args.block_stats:
                stats_list = _block_size_stats(cnf_path, n)
                edges_list: List[int] = []
                if args.complete and args.backend == "networkx":
                    edges_list = _block_edge_list_networkx(cnf_path, n)
                print("block_idx\titer\tcutwidth\tclauses\tvars\tlits\tedges")
                for i, (lab, v) in enumerate(blocks):
                    s = stats_list[i] if i < len(stats_list) else {}
                    edges = edges_list[i] if i < len(edges_list) else 0
                    print(
                        f"{i}\t{lab}\t{v}\t"
                        f"{int(s.get('clauses', 0))}\t{int(s.get('vars', 0))}\t{int(s.get('lits', 0))}\t{int(edges)}"
                    )
            else:
                for lab, v in blocks:
                    print(f"{lab}\t{v}")
        return

    # random trials: keep best (smallest) as approximation of optimal cutwidth
    best_val = None
    nx_graph = None
    nx_block_graphs = None
    if args.complete and args.backend == "networkx":
        # Build once and reuse across trials.
        nx_graph = _build_primal_graph_nx(cnf_path, n)
        if args.mode == "block":
            nx_block_graphs = _build_primal_graph_blocks_nx(cnf_path, n)
    for t in range(args.trials):
        pos = build_pos_map(n, "random", args.seed, t)
        if args.mode == "block":
            if args.complete:
                if args.backend == "networkx":
                    assert nx_block_graphs is not None
                    blocks = [
                        (
                            lab,
                            _cutwidth_from_edge_positions(
                                n, ((pos[u], pos[v]) for u, v in g.edges())
                            ),
                        )
                        for lab, g in nx_block_graphs
                    ]
                else:
                    blocks = primal_cutwidth_complete_blocks_under_pos(cnf_path, n, pos)
            else:
                blocks = primal_cutwidth_blocks_under_pos(
                    cnf_path, n, pos,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )
            if not blocks:
                if args.complete:
                    if args.backend == "networkx":
                        assert nx_graph is not None
                        val = _cutwidth_from_edge_positions(
                            n, ((pos[u], pos[v]) for u, v in nx_graph.edges())
                        )
                    else:
                        val = primal_cutwidth_complete_under_pos(cnf_path, n, pos)
                else:
                    val = primal_cutwidth_under_pos(
                        cnf_path, n, pos,
                        clause_cap=args.clause_cap,
                        skip_large=args.skip_large,
                    )
                if best_val is None or val < best_val:
                    best_val = val
            else:
                # For random ordering + block mode, we report per-trial results:
                # label trial:<t>:<iter>
                for lab, v in blocks:
                    print(f"trial:{t}:{lab}\t{v}")
        else:
            if args.complete:
                if args.backend == "networkx":
                    assert nx_graph is not None
                    val = _cutwidth_from_edge_positions(
                        n, ((pos[u], pos[v]) for u, v in nx_graph.edges())
                    )
                else:
                    val = primal_cutwidth_complete_under_pos(cnf_path, n, pos)
            else:
                val = primal_cutwidth_under_pos(
                    cnf_path, n, pos,
                    clause_cap=args.clause_cap,
                    skip_large=args.skip_large,
                )
            if best_val is None or val < best_val:
                best_val = val

    if args.mode != "block":
        print(best_val if best_val is not None else 0)


if __name__ == "__main__":
    main()
