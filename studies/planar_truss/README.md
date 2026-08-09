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

For diagnostics with the historical loss, the runner also exposes
`--hyperparameter-selection cross_validation`. This deterministic five-fold
selector uses only held-out responses from the current DOE and never the
reference failure probability.

Audit a frozen confirmation campaign and regenerate the state-of-the-art
comparison table with:

```bash
python -m studies.planar_truss.audit_campaign
```

The committed version-2 confirmation uses ten algorithm seeds held out from
the three development seeds, the gradient term disabled, a fixed candidate
population of `2^17`, and 95 OpenSees calls per run. The 95-call budget was
selected in a clearly labeled reference-aware development analysis before the
confirmation protocol was committed; neither the subsequent training loops
nor model selection use the reference or validation samples. The confirmed
mean is `Pf = 1.47340e-3`, with 2.310% relative error against the independent
16-replication RQMC reference and 3.066% against the published direct-MCS
value. The median and maximum per-run errors are 1.549% and 7.667%.

All ten runs remain in `confirmation_campaign_v2.json`. The automated audit
checks the precommitted seed split, loss, response, call budget, candidate
population, evidence schedule, and validation design. Compared with published
results, this campaign uses fewer true-model calls than A-bPCE (95 versus 129)
but has slightly higher error against the independent RQMC reference (2.310%
versus 1.872%); the published ABSVR1/2 estimates remain more efficient and
more accurate on this benchmark.

## Direct UQLab comparator

The `uqlab/` directory contains direct UQLab reliability configurations whose
true model evaluations are sent to the same OpenSeesPy implementation through
a tested CSV bridge. Each run has an independent OpenSees call ledger, and the
runner rejects any disagreement with UQLab's own `ModelEvaluations` count.

See `uqlab/README.md` for installation and execution. The confirmed comparator
uses UQLab 2.1.0's native `AKMCS` method, Gaussian Kriging, the U learning
function, and its original `min(U) >= 2` stopping rule. Ten algorithm seeds were
committed before confirmation; the development seed was excluded. Every run
uses 30 initial LHS points and an internal MCS population of one million.

The confirmed UQLab mean is `Pf = 1.52560e-3`, with 1.151% relative error
against the independent RQMC reference. It requires 383.4 OpenSees calls on
average (range 334--409). ABSVR's confirmed mean uses 95 calls, a 75.222%
reduction, while its mean absolute per-run error is only 6.223% higher than the
direct UQLab comparator (2.641% versus 2.487%). The separate 95% intervals
across algorithm seeds are recorded in `comparison_audit_v3.json`.

This is a direct, reproducible UQLab comparison on the same solver and
probabilistic model, not an exact reproduction of the single published AK-MCS
realization: the 2018 paper does not specify enough information to reconstruct
its uniform-in-a-ball initial design, so the local campaign uses a disclosed
LHS. A-bPCE remains a published external comparator from a different surrogate
family, not a modification of ABSVR.
