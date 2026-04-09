BMC categorization pipeline

## Purpose

Classify BMC instances as **linear**, **polynomial**, or **exponential** in solving-time growth. The pipeline unrolls circuits into CNF, runs a SAT solver, collects solving times, fits regression models to the running-maximum curve, and selects the best-fitting label.

## Workflow

From the repository root, set the import path once:
```bash
export PYTHONPATH=src/scripts
```

**1. Generate CNFs** — unroll each BMC circuit to depths K = 1…100 and write one CNF per depth.
```bash
python src/scripts/prepare_formulas.py --name <instance> --k_limit 100 --time_limit 1600
```

**2. Collect solving times** — run CaDiCaL on the generated CNFs and record solving time per depth (timeout: 1600 s).
```bash
python src/scripts/Experiments/collect_solving_time.py --all
```

**3. Compute running-max and interpolate** — replace each solving time with the running maximum up to that depth (monotone non-decreasing). The curve is then smoothed by piecewise-linear interpolation: knot points are placed wherever the running max rises, and values between knots are linearly interpolated. This produces a smooth series suitable for regression.

done in Experiments/direct_regression_analysis.py

**4. Fit regressions** — fit linear, polynomial, and exponential models to the smoothed series. The model with the **highest R²** is selected; linear receives a constant **+0.05 bonus** to break near-ties in its favour. Steps 3 and 4 are both performed by:
```bash
python src/scripts/Experiments/direct_regression_analysis.py --all --output regression.json --summary ''
```

For CSV schemas and output formats, see [`src/scripts/`](src/scripts/) or open an issue.
