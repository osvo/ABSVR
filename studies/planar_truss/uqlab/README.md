# Direct UQLab AK-MCS comparison

This directory connects UQLab directly to the same OpenSeesPy limit-state
model used by ABSVR. UQLab owns the input sampling, Kriging fit, learning
function, stopping rule, and failure-probability estimate. The MATLAB model
wrapper sends every true evaluation to OpenSeesPy in a CSV batch.

Every bridge call is recorded in an append-only JSONL ledger. A run is rejected
if the ledger count differs from `analysis.Results.ModelEvaluations` reported by
UQLab. Surrogate evaluations in the internal Monte Carlo population do not count
as true limit-state calls.

## Installation prerequisite

UQLab 2.x is BSD-3 licensed and needs no license manager, but ETH Zurich
requires a free website account to download its source. Download and install it
from <https://www.uqlab.com/download>, follow
<https://www.uqlab.com/install>, and run `uqlab -selftest`. Do not commit UQLab
itself to this repository.

MATLAB must see UQLab's `core` directory and the repository virtual environment
must contain OpenSeesPy. If needed, point the wrapper to another interpreter:

```matlab
setenv('ABSVR_PYTHON', 'C:\path\to\python.exe')
```

## One independently seeded replication

From MATLAB, after adding UQLab and this directory to the path:

```matlab
addpath('C:\path\to\UQLab\core')
addpath(fullfile(pwd, 'studies', 'planar_truss', 'uqlab'))
run_uqlab_akmcs(11)
```

The default `paper_like` profile explicitly uses:

- physical lognormal and Gumbel marginals from the published truss;
- 30 initial LHS points;
- Gaussian-covariance Kriging;
- the U learning function;
- `StopPfBound` with a 10% threshold for two consecutive iterations;
- an internal MCS population and maximum candidate population of one million.

The 2018 A-bPCE article instead initialized its truss study with a uniform
sample in a ball, without enough information in the article to reconstruct the
ball uniquely. Consequently, this is a direct, disclosed UQLab baseline and a
paper-like reconstruction, not an exact repetition of the reported 300-call
AK-MCS realization. The published AK-MCS and A-bPCE values remain separate in
`../literature_results.csv`.

For the original AK-MCS `min(U) >= 2` stopping condition, use:

```matlab
run_uqlab_akmcs(11, "", "original_akmcs")
```

Each run writes a JSON summary, a MATLAB result file, and the OpenSees call
ledger under `results/planar_truss/`. Record all seeds before starting a
confirmation campaign; never select or discard runs based on their error.

