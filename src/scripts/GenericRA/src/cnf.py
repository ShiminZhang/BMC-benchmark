import os
import json
from .logging import LOG, LOG_TAG
from tqdm import tqdm
import argparse
import numpy as np
import re


class CNF:
    def __init__(self,cnf_path=None):
        self.cnf_path = cnf_path
        self.clauses = []
        self.N = None
        self.L = None
        self.iter_map = {}
        self.literal_set = set()
        self.K = -1
        if cnf_path is not None:
            self.parse_cnf()
    
    @classmethod
    def from_file(cls, cnf_path):
        return cls(cnf_path)
    
    def parse_cnf(self):
        with open(self.cnf_path, 'r') as file:
            line_count = 0
            self.iter_map[0] = 0
            iter_count = 1
            for line in file:
                if line.startswith('p cnf'):
                    _, _, L , N = line.split()
                    self.N = int(N)
                    self.L = int(L)
                elif line.startswith('c'):
                    if line.startswith('c iter'):
                        self.iter_map[iter_count] = line_count
                        iter_count += 1
                    continue
                elif line.startswith('v'):
                    continue
                else:
                    literals = [int(x) for x in line.strip().split() if x != '0']
                    if literals:  # Only add non-empty clauses
                        self.clauses.append(literals)
                        line_count += 1
            self.K = iter_count - 1
            assert self.N is not None and self.L is not None
            LOG(f"N: {self.N}, L: {self.L}")
            LOG(f"Number of clauses: {len(self.clauses)}")
            self.parse_literals()

    def dump_stats(self):
        print(f"N: {self.N}, L: {self.L}")
        print(f"Number of clauses: {len(self.clauses)}")
        print(f"Number of literals: {len(self.literal_set)}")
        print(f"Number of unique literals: {len(self.literal_set)}")
        print(f"Number of clauses: {len(self.clauses)}")
        print(self.iter_map)

    def append_clause(self, clause):
        if not isinstance(clause, list):
            raise TypeError("Clause must be a list of integers")
        if not all(isinstance(x, int) for x in clause):
            raise TypeError("All elements in clause must be integers")
        if not clause:
            raise ValueError("Clause cannot be empty")
        self.clauses.insert(0,clause)
        self.N += 1
        self.L = max(self.L, max(abs(literal) for literal in clause))
        self.parse_literals()
        return self
    
    def parse_literals(self):
        assert self.clauses is not None
        for clause in self.clauses:
            for literal in clause:
                self.literal_set.add(literal)
    
    def get_clauses(self):
        return self.clauses
    
    def get_N(self):
        return self.N
    
    def get_L(self):
        return self.L
    
    def get_clause_at(self, index):
        return self.clauses[index]
    
    def get_iter_map(self):
        return self.iter_map
    
    def get_literals(self):
        return self.literal_set
    
    def init_with_clauses(self,clauses):
        LOG_TAG(f"init_with_clauses: {clauses}", "detailed")
        self.clauses = clauses
        self.N = len(clauses)
        if len(clauses) == 0:
            self.L = 0   
            self.iter_map = {}
            return         
        self.L = max(max(abs(literal) for literal in clause) for clause in clauses if len(clause) > 0)
        self.iter_map = {}
        self.parse_literals()
    
    def get_A(self, i):
        clauses = self.clauses[0:self.iter_map[i+1]]
        A = CNF()
        A.init_with_clauses(clauses)
        return A
    
    def get_B(self, i):
        clauses = self.clauses[self.iter_map[i+1]:]
        B = CNF()
        B.init_with_clauses(clauses)
        return B
    
    def to_dimacs(self, file_path):
        """Write the CNF formula to a file in DIMACS format."""
        with open(file_path, 'w') as f:
            # Write header
            f.write(f"p cnf {self.L} {self.N}\n")
            # Write clauses
            for clause in self.clauses:
                f.write(" ".join(str(lit) for lit in clause) + " 0\n")
        return file_path

def GetDataFromLog(log_path):
    with open(log_path, 'rb') as file:
        file.seek(0, 2)
        position = file.tell()
        line = b''
        linecnt=0
        phase=0 # 0 for time, 1 for mem
        while position >= 0 and linecnt <= 500:        
            file.seek(position)
            char = file.read(1)
            if char == b'\n' and line:
                linecnt+=1
                decoded_line = line.decode('utf-8')
                if "raising signal" in decoded_line:
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {log_path}")
                    continue
                if "mylog" in decoded_line:
                    continue

                if "process-time" in decoded_line or "total process time" in decoded_line:
                    match = re.search(r'(\d+\.?\d*)\s+seconds', decoded_line) or re.search(r'total process time[^:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds', decoded_line)
                    if match:
                        # print(basename)
                        time = float(match.group(1))
                        return time
                    
                if "CPU time" in decoded_line in decoded_line:
                    match = re.search(r'CPU time[^:]*:\s*([0-9]+(?:\.[0-9]+)?)\s*s', decoded_line)
                    if match:
                        # print(basename)
                        time = float(match.group(1))
                        return time
                line = b''
            else:
                line = char + line
            position -= 1
    return None