# ABSVR: Adaptive Bayesian Support Vector Regression

Surrogate-based structural reliability analysis using adaptive Bayesian SVR with active learning.

## Overview

ABSVR estimates the **probability of failure** of engineering systems by training an SVR surrogate on a small set of limit state function evaluations and iteratively enriching it through active learning. The algorithm:

1. Generates an initial Latin Hypercube design in standard normal space
2. Trains a Bayesian SVR surrogate with automatic hyperparameter optimization via evidence maximization
3. Scores candidates using a composite learning function that balances exploitation (proximity to the failure boundary) with exploration (model uncertainty, probability density, and gradient information)
4. Evaluates the most informative candidate on the true limit state function
5. Repeats until the reliability index stabilizes and the Monte Carlo pool resolution is sufficient

## Installation

```bash
git clone https://github.com/osvo/ABSVR.git
cd ABSVR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+.

The finite-element benchmark has an optional OpenSeesPy dependency:

```bash
pip install -r requirements-structural.txt
```

## Usage

```bash
# Interactive mode
python -m absvr_cli

# Non-interactive
python -m absvr_cli --benchmark eg2 --random-seed 42

# Published 23-bar truss evaluated by OpenSeesPy
python -m absvr_cli --benchmark eg7 --random-seed 42
```

Full options: `python -m absvr_cli --help`

## Benchmarks

Seven reliability benchmarks are included for validation. Example 7 is an
implicit finite-element problem; the other six are analytical limit states.

| ID  | Problem              | Dims | Inputs    | Reference Pf  |
|-----|----------------------|------|-----------|---------------|
| eg1 | Highly nonlinear     | 2    | Normal    | 4.709e-3      |
| eg2 | Four-branch series   | 2    | Normal    | 2.220e-3      |
| eg3 | Nonlinear oscillator | 6    | Normal    | 2.861e-2      |
| eg4 | Modified Rastrigin   | 2    | Normal    | 7.296e-2      |
| eg5 | 2D decision function | 2    | Normal    | 1.851e-3      |
| eg6 | High-dimensional     | 40   | Lognormal | 5.080e-3      |
| eg7 | 23-bar planar truss  | 10   | Mixed via Nataf transform | 1.520e-3 |

Example 7 is the simply supported 23-bar, 13-node truss used by Schobi et al.
(2016), Marelli and Sudret (2018), and Wang et al. (2021). Its four section
and material variables are lognormal, its six loads are Gumbel, and failure is
defined by a 0.12 m midspan-displacement limit. The benchmark implementation
uses OpenSeesPy for every true limit-state call and includes an independent
NumPy finite-element oracle for verification.

Validation results (100 seeds per benchmark) are available in `results/`.

## Project Structure

```
ABSVR/
├── absvr_core/                # Core adaptive learning loop
│   ├── adaptive_loop.py       #   Main algorithm orchestration
│   ├── learning_function.py   #   Composite candidate scoring (SLF)
│   ├── candidate_selection.py #   Best-candidate selection and DOE enrichment
│   ├── ood_scoring.py         #   Tail-sensitive out-of-distribution adjustment
│   ├── probability_estimation.py
│   ├── problem_setup.py       #   Benchmark loader and custom problem support
│   ├── checkpointing.py       #   State persistence (npz format)
│   └── profiles.py            #   Baseline hyperparameter profiles
├── surrogate/svr/             # Bayesian SVR implementation
│   ├── train.py               #   Data-driven defaults and training dispatcher
│   ├── model.py               #   Evidence maximization via boxmin
│   ├── train_internal.py      #   Epsilon-SVR dual QP solver (CVXOPT / SciPy)
│   ├── predict.py             #   Predictive mean and variance
│   ├── kernel_utils.py        #   RBF, Laplacian, Polynomial kernels
│   ├── ensemble.py            #   Bagged SVR ensemble
│   └── grad.py                #   Analytical gradient of the predictive mean
├── benchmarks/                # Limit state function definitions
├── sampling/                  # Sobol quasi-random sequence generation
├── utilities/                 # LHS, random stream, memory profiling
├── results/                   # 100-seed validation CSVs (eg1 to eg6)
├── absvr_cli.py               # CLI entry point
├── requirements.txt
└── LICENSE
```

## License

MIT. See [LICENSE](LICENSE).
