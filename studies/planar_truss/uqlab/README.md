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

The development-only `paper_like` profile explicitly uses:

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

That profile is not used for confirmation: a seed-127 precheck stopped after
only one adaptive point with `Pf = 0`. The degenerate result is retained under
`results/planar_truss/uqlab_precheck/` and marked rejected in
`uqlab_precheck_decisions.json`.

For UQLab's native implementation of the original AK-MCS
`min(U) >= 2` stopping condition, use:

```matlab
run_uqlab_akmcs(11, "", "original_akmcs")
```

This profile selects UQLab's dedicated `Method = 'AKMCS'` and
`AKMCS.Convergence = 'stopU'`. It does not emulate the rule through the newer
ALR `StopLF` interface, whose signed learning-function convention is different.

Each run writes a JSON summary, a MATLAB result file, and the OpenSees call
ledger under `results/planar_truss/`. Record all seeds before starting a
confirmation campaign; never select or discard runs based on their error.

For the frozen ten-seed native-AK-MCS campaign, run from the repository root:

```bash
python -m studies.planar_truss.run_uqlab_campaign \
  --uqlab-core C:\path\to\UQLab\core \
  --output-directory results/planar_truss/uqlab_confirmation --resume
```

The Python orchestrator starts an independent MATLAB process for each declared
seed, writes the protocol before the first run, validates every ledger, and
refuses to aggregate an incomplete or reordered seed set. `--resume` only skips
a result after checking that its seed, profile, and call counts match its
campaign slot.
