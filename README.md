# BMC-benchmark
Benchmark study of bounded model checking (BMC) instances from the HWMCC15–19 collection.

Goal: discriminate clearly linear instances from clearly exponential 

## Overview
- Instances are provided as AIG circuits under `data/aigs/`.
- Each circuit is incrementally unrolled until the corresponding CNF requires more than 1800 s for CaDiCaL to solve.
- Scaling behaviour is characterized by combining solver logs, symbolic regression, and manual inspection to classify growth as linear, polynomial, exponential, or unknown.

## Methodology
1. **CNF generation and solving**
   - Unroll each circuit frame-by-frame.
   - Call CaDiCaL on the generated CNF.
   - Stop once the solve exceeds the 1800 s timeout.
2. **Symbolic regression**
   - Fit solve time as a function of CNF clause count using `pysr`.
   - Simplify the resulting expressions with `sympy`.
3. **Asymptotic interpretation**
   - Use Gemini-2.5-Flash to infer asymptotic growth from the simplified expressions.
   - Validate Pysr and GPT outputs through manual review of plots and expressions.
4. **Visualization**
   - Plot clause count (x-axis) versus solve time (y-axis) for each instance.
   - Inspect `results/plots/*original_only.png` for the visualization of each instance.

## Results
- Final curated labels and supporting data are stored in `report_checked.csv`.
- Solver logs, including hardware metadata and timestamps, are available under `results/solving_logs/`.
- Arguments from LLM of upper bounds are in results/conclusion/
- A condensed view of the conclusions is published at: https://docs.google.com/spreadsheets/d/1L0ndBIWAwAvGYVo3usdulh3baRSvy_p9cTdlEUNFt6U/edit?usp=sharing

## Classification Criteria
- Goal: discriminate clearly linear instances from clearly exponential ones.
- Assignments:
  - Linear: clearly linear scaling.
  - Exponential: clearly exponential scaling.
  - Polynomial: sub-exponential growth that is not convincingly linear.
  - Unknown: ambiguous trend, insufficient pattern, or behaviour between polynomial and exponential. 
  - Too few data: should be treated the same as unknown. This includes Exponential(too few data)

## Additional Notes
- Missing or malformed or failed-to-parse or out-of-token LLM outputs remain `NA` in `report_checked.csv`; manual review ensures these omissions do not affect the labels correctness.
