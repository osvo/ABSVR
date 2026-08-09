# Published 23-bar planar-truss benchmark

This study reproduces the simply supported Warren truss used as a structural
reliability benchmark by Schobi et al. (2017), Marelli and Sudret (2018), and
Wang et al. (2021). The structure has 13 nodes, 23 linear truss elements, a
24 m span, and a 2 m height. Six independent vertical loads act on the upper
nodes. Failure is a downward midspan displacement of at least 0.12 m.

## Published probabilistic model

| Variables | Distribution | Mean | Standard deviation |
|---|---:|---:|---:|
| E1, E2 (Pa) | Lognormal | 2.1e11 | 2.1e10 |
| A1 (m2) | Lognormal | 2.0e-3 | 2.0e-4 |
| A2 (m2) | Lognormal | 1.0e-3 | 1.0e-4 |
| P1, ..., P6 (N) | Gumbel | 5.0e4 | 7.5e3 |

All variables are independent. ABSVR samples independent standard-normal
coordinates and the benchmark applies the corresponding isoprobabilistic
transform before each finite-element evaluation.

## Traceable sources

- Schobi, Sudret, and Marelli (2017), *Rare Event Estimation Using
  Polynomial-Chaos Kriging*, DOI: 10.1061/AJRUA6.0000870.
- Marelli and Sudret (2018), *An active-learning algorithm that combines
  sparse polynomial chaos expansions and bootstrap for structural reliability
  analysis*, DOI: 10.1016/j.strusafe.2018.06.003.
- Wang et al. (2021), *Efficient structural reliability analysis based on
  adaptive Bayesian support vector regression*, DOI:
  10.1016/j.cma.2021.114172.

The published reference is `Pf = 1.52e-3` from one million direct Monte Carlo
evaluations. `literature_results.csv` records the comparison values reported in
Table 8 of Wang et al. (2021).

## Reproduction

Install the optional finite-element dependency and run the verification:

```bash
pip install -r requirements-structural.txt
python -m studies.planar_truss.verify_model
pytest -q tests/test_planar_truss_opensees.py
```

Estimate an independent randomized-QMC reference using 16 Sobol scrambles of
2^20 samples each:

```bash
python -m studies.planar_truss.run_reference_qmc \
  --log2-samples 20 --replications 16 \
  --output results/planar_truss/reference_qmc.json
```

The direct reference calculation uses a vectorized virtual-work expression
derived from the same truss topology. It is not used by ABSVR. Every true
limit-state call made by the `eg7` benchmark is evaluated with OpenSeesPy.

Run the repeated ABSVR campaign and its gradient-term ablation with:

```bash
python -m studies.planar_truss.run_absvr_campaign
```

The default campaign uses ten algorithm seeds, 15 initial plus 80 adaptive
OpenSees evaluations per run, gradient weights 0 and 1, and the explicitly
declared squared epsilon-insensitive loss. Its likelihood normalizer,
curvature determinant, and dual QP therefore belong to the same loss model.
Hyperparameters are selected periodically by Bayesian evidence using only the
current design. Independent scrambled Sobol samples are used only after each
model has been frozen. Their reference evaluations are reported explicitly
and are not counted as calls made by ABSVR.

Audit a frozen confirmation campaign and regenerate the state-of-the-art
comparison table with:

```bash
python -m studies.planar_truss.audit_campaign
```

The committed ten-seed confirmation uses the reference-blind protocol with
the gradient term disabled, a fixed candidate population of `2^17`, and 95
OpenSees calls per run. Its mean estimate is `Pf = 1.50135e-3`: the relative
error is 0.457% against the independent 16-replication RQMC reference and
1.227% against the published direct-MCS value. Per-run errors and the two
least accurate seeds remain in the result file and audit; no run was removed.

## Direct UQLab comparator

The `uqlab/` directory contains a direct UQLab ALR/AK-MCS configuration whose
true model evaluations are sent to the same OpenSeesPy implementation through
a tested CSV bridge. Each run has an independent OpenSees call ledger, and the
runner rejects any disagreement with UQLab's own `ModelEvaluations` count.

See `uqlab/README.md` for installation and execution. The default local profile
matches the published marginals, initial size, Gaussian Kriging covariance,
internal MCS size, and 10% failure-probability-bound stopping rule. It uses a
fully disclosed LHS because the 2018 paper does not specify enough detail to
reconstruct its uniform-in-a-ball initial design exactly. It is therefore
reported as a direct paper-like UQLab baseline, not as an exact reproduction of
the published AK-MCS realization. A-bPCE remains a published external
comparator rather than a modification of ABSVR.
