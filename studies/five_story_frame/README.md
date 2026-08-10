# Five-storey structural-frame reliability study

This study reproduces the three-bay, five-storey frame used by Blatman and
Sudret (2010), Marelli and Sudret (2018), and Wagner et al. (2022).  It uses
21 correlated inputs, a Gaussian copula, and the top-right horizontal
displacement limit state `g = 0.05 - u` in SI units.

The production limit-state function calls OpenSeesPy.  An independent NumPy
assembly and an exact symmetric-banded solver verify the model and make a
large direct reference simulation feasible.  The banded solver is not a
surrogate: it assembles the same 60-degree-of-freedom elastic stiffness matrix
from 16 unit-property bases and solves it with LAPACK.

Run the independent randomized quasi-Monte Carlo reference with:

```powershell
.\.venv\Scripts\python.exe -m studies.five_story_frame.run_reference_rqmc
```

The default protocol uses 16 independently scrambled Sobol replicates of
`2^17` points.  The replicate dispersion, rather than a binomial formula for
pseudorandom Monte Carlo, defines the reported uncertainty interval.

## Literature checkpoints

- Marelli and Sudret (2018) report `Pf = 1.54e-3` with a 95% importance-
  sampling interval `[1.51e-3, 1.56e-3]` at the 5 cm threshold.  They report
  A-bPCE at `1.49e-3` using 235 model calls.
- Li et al. (2019) report direct-MCS displacement moments of 0.0652 ft and
  0.0202 ft from 100,000 evaluations.  These agree with this implementation.
- Wagner et al. (2022) use the same dependent input model at a 9 cm threshold
  and report a direct-MCS reference `Pf = 1.49e-6` from `10^8` evaluations.

The older 0.069 ft quadrature mean quoted alongside the Li et al. comparison
does not agree with that paper's own direct MCS (0.0652 ft).  It is retained as
a documented source discrepancy and is not used to rescale or calibrate the
finite-element response.

## Sources

- S. Marelli and B. Sudret, *Structural Safety* 75 (2018), 67-74,
  <https://doi.org/10.1016/j.strusafe.2018.06.003>.
- C. Li et al., *Engineering Computations* 35(6) (2018), 2165-2214,
  <https://doi.org/10.1108/EC-04-2017-0140>.
- P.-R. Wagner et al., *Structural Safety* 96 (2022), 102179,
  <https://doi.org/10.1016/j.strusafe.2021.102179>.

