import re
import random
from typing import Dict, List, Optional, Tuple


class ScrambleType:
    CLAUSE = "clause"
    CLAUSE_AND_ITER = "clauseiter"


SCRAMBLE_TYPES = [ScrambleType.CLAUSE, ScrambleType.CLAUSE_AND_ITER]

# name__scr_<type>_s<seed>  (seed may be negative)
_SCRAMBLED_NAME_RE = re.compile(r"^(?P<name>.+)__scr_(?P<type>[a-zA-Z]+)_s(?P<seed>-?\d+)$")


def make_scrambled_name(name: str, scramble_type: str, seed: int) -> str:
    return f"{name}__scr_{scramble_type}_s{seed}"


def parse_scrambled_name(virtual_name: str) -> Optional[Tuple[str, str, int]]:
    match = _SCRAMBLED_NAME_RE.match(virtual_name)
    if not match:
        return None
    return match.group("name"), match.group("type"), int(match.group("seed"))


class SimpleCNF:
    """Minimal DIMACS CNF reader/writer, ported from ProofDoorTools/scripts/utils/process_cnf.py.

    Only keeps what scrambling needs: clause list + iteration boundaries
    (`c iter` comments). Unlike process_cnf.CNF, this has no dependency on
    the top-level ProofDoorTools `utils` package (which collides in name
    with BMCBenchmark's own `utils` package).
    """

    def __init__(self):
        self.clauses: List[List[int]] = []
        self.iter_map: Dict[int, int] = {}
        self.nvar = 0
        self.nclause = 0

    @classmethod
    def from_file(cls, cnf_path: str) -> "SimpleCNF":
        cnf = cls()
        clauses: List[List[int]] = []
        iter_map: Dict[int, int] = {0: 0}
        iter_count = 1
        nvar = None
        nclause = None
        with open(cnf_path, "r") as f:
            for line in f:
                if line.startswith("p cnf"):
                    _, _, nvar_s, nclause_s = line.split()
                    nvar, nclause = int(nvar_s), int(nclause_s)
                elif line.startswith("c"):
                    if line.startswith("c iter"):
                        iter_map[iter_count] = len(clauses)
                        iter_count += 1
                    continue
                elif line.startswith("v"):
                    continue
                else:
                    literals = [int(x) for x in line.strip().split() if x != "0"]
                    if literals:
                        clauses.append(literals)
        cnf.clauses = clauses
        cnf.iter_map = iter_map
        cnf.nvar = nvar if nvar is not None else 0
        cnf.nclause = nclause if nclause is not None else len(clauses)
        return cnf

    def to_dimacs(self, output_path: str) -> str:
        boundary_at = {start: iter_idx for iter_idx, start in self.iter_map.items()}
        with open(output_path, "w") as f:
            f.write(f"p cnf {self.nvar} {len(self.clauses)}\n")
            for clause_idx, clause in enumerate(self.clauses):
                if clause_idx in boundary_at:
                    f.write(f"c iter {boundary_at[clause_idx]} \n")
                f.write(" ".join(str(lit) for lit in clause) + " 0\n")
        return output_path


def _build_iter_blocks(cnf: SimpleCNF) -> List[List[List[int]]]:
    clauses = cnf.clauses
    iter_map = cnf.iter_map
    if not iter_map:
        return [clauses]
    iter_keys = sorted(iter_map.keys())
    blocks = []
    for idx, iter_idx in enumerate(iter_keys):
        start = iter_map[iter_idx]
        end = len(clauses) if idx + 1 == len(iter_keys) else iter_map[iter_keys[idx + 1]]
        blocks.append(clauses[start:end])
    return blocks


def _cnf_from_blocks(blocks: List[List[List[int]]], nvar: int) -> SimpleCNF:
    new_clauses = [clause for block in blocks for clause in block]
    new_cnf = SimpleCNF()
    new_cnf.clauses = new_clauses
    new_cnf.nvar = nvar
    new_cnf.nclause = len(new_clauses)
    new_iter_map = {}
    clause_index = 0
    for iter_index, block in enumerate(blocks):
        new_iter_map[iter_index] = clause_index
        clause_index += len(block)
    new_cnf.iter_map = new_iter_map
    return new_cnf


def scramble_cnf_clauses(cnf: SimpleCNF, rng: random.Random) -> SimpleCNF:
    # shuffle all clauses across the whole formula; iter_map boundary
    # *positions* are kept as-is, but which clause sits at a given position
    # changes, so the boundaries no longer align with the original iterations.
    clauses = cnf.clauses[:]
    rng.shuffle(clauses)
    new_cnf = SimpleCNF()
    new_cnf.clauses = clauses
    new_cnf.nvar = cnf.nvar
    new_cnf.nclause = cnf.nclause
    new_cnf.iter_map = cnf.iter_map
    return new_cnf


def scramble_cnf_clause_and_iteration(cnf: SimpleCNF, rng: random.Random) -> SimpleCNF:
    # shuffle iteration blocks, then shuffle clauses inside each block
    blocks = _build_iter_blocks(cnf)
    rng.shuffle(blocks)
    for block in blocks:
        rng.shuffle(block)
    return _cnf_from_blocks(blocks, cnf.nvar)


def scramble_cnf_file(input_path: str, output_path: str, scramble_type: str, rng: random.Random) -> str:
    cnf = SimpleCNF.from_file(input_path)
    if scramble_type == ScrambleType.CLAUSE:
        scrambled = scramble_cnf_clauses(cnf, rng)
    elif scramble_type == ScrambleType.CLAUSE_AND_ITER:
        scrambled = scramble_cnf_clause_and_iteration(cnf, rng)
    else:
        raise ValueError(f"Invalid scramble type: {scramble_type}")
    scrambled.to_dimacs(output_path)
    return output_path
