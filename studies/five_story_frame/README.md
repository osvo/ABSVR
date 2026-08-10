# Five-storey structural-frame reliability study

This study reproduces the three-bay, five-storey frame used by Blatman and
Sudret (2010), Marelli and Sudret (2018), and Wagner et al. (2022).  It uses
21 correlated inputs, a Gaussian copula, and the top-right horizontal
displacement limit state `g = 0.05 - u` in SI units.

The production limit-state function calls OpenSeesPy's
`ElasticTimoshenkoBeam`.  The source papers say only that their in-house model
is a linear finite-element model; they do not document its shear-deformation
convention.  We therefore fix, before reliability estimation, the conventional
values `nu = 0.30` and `As = 5A/6`.  This reconstruction reproduces the
published 0.069 ft and 0.021 ft response moments without scaling the response.

An independent dense NumPy assembly and an exact symmetric-banded solver
verify the OpenSees model and make a large direct reference simulation
feasible.  The banded solver is not a surrogate: it assembles and solves the
same 60-degree-of-freedom Timoshenko stiffness matrix with LAPACK.  A complete
Euler-Bernoulli sensitivity model is retained separately; it reproduces Li et
al.'s alternate direct-MCS moments of 0.0652 ft and 0.0202 ft.

Run the independent randomized quasi-Monte Carlo reference with:

```powershell
.\.venv\Scripts\python.exe -m studies.five_story_frame.run_reference_rqmc
```

The default protocol uses 16 independently scrambled Sobol replicates of
`2^17` points.  The replicate dispersion, rather than a binomial formula for
pseudorandom Monte Carlo, defines the reported uncertainty interval.

Run the three-seed ABSVR development campaign with:

```powershell
.\.venv\Scripts\python.exe -m studies.five_story_frame.run_absvr_campaign
```

Its fixed starting profile uses a 43-point normal LHS (`2M+1` for 21 inputs),
80 single-point enrichments, squared epsilon-insensitive loss, and a `2^17`
candidate population.  The reference probability and the post-training
validation set are unavailable to the adaptive loop.  All OpenSees calls in
the initial design and enrichment sequence count toward the reported budget;
the separately labeled reference audit does not.

The first three-seed development profile (`SLF`, 123 calls) was rejected: its
mean probability was `1.2051e-3`, giving 20.20% relative error, with individual
errors from 2.05% to 42.15%.  The single favorable seed is not selected or
reported as representative.  The raw campaign remains in
`results/five_story_frame/absvr_development.json`.

Replacing SLF by `u_distance` at the same 123-call budget reduced the error of
the three-seed mean to 13.25%, but individual errors still ranged from 1.86%
to 30.34%.  This profile is also rejected at that budget; the complete result
is retained in `results/five_story_frame/u_distance_development.json`.

Extending the same three reference-aware development paths to 235 calls (the
published A-bPCE budget) produced individual errors of 0.095%, 0.600%, and
4.642% against the independent RQMC reference.  The error of the mean estimate
was 1.779%.  This is a candidate profile, not a confirmation result; the
extension decision used the preceding development results and is disclosed in
`results/five_story_frame/u_distance_budget235_development.json`.

A predeclared prefix rule then selected the smallest stored budget with at
most 3% error in the development mean and at most 5% error in every
development seed.  The selected 143-call prefix produced individual errors of
0.158%, 0.537%, and 1.295%, and 0.200% error in the mean.  Prefix reconstruction
used the original evidence-retuning schedule and required no additional
structural calls.  This is reference-aware budget selection and remains
development evidence only until the frozen, disjoint-seed confirmation passes.

The frozen 143-call profile failed confirmation on ten disjoint seeds.  Its
mean estimate had 6.09% error, the mean absolute error across runs was 15.44%,
and the worst run had 40.70% error.  The profile is rejected; the favorable
three-seed development result is not used as manuscript evidence.  The full
failed confirmation is retained in
`results/five_story_frame/confirmation_campaign.json`.

Those ten exposed seeds were then reclassified as development and extended to
235 calls to test the stated stability hypothesis.  At 235 calls, the error of
the ten-seed mean fell to 0.733%, the mean absolute run error to 4.16%, and the
worst run error to 9.13%.  No intermediate prefix was selected after seeing
these results: the 235-call endpoint is carried forward to a new confirmation
on a second disjoint seed set.  The development record is
`results/five_story_frame/postconfirmation_budget235_development.json`.

The frozen second confirmation passed on ten entirely new seeds.  Every run
used 235 OpenSees calls.  The confirmed mean was `1.51711e-3`, with 0.461%
error against the independent RQMC reference; the mean absolute run error was
3.47%.  Eight runs had errors between 0.60% and 1.67%, while two outlying runs
had 6.88% and 18.00% error.  All ten runs and their dispersion must be reported,
not only the mean or the favorable subset.  The immutable record is
`results/five_story_frame/confirmation_campaign_v2.json`.

## Literature checkpoints

- Marelli and Sudret (2018) report `Pf = 1.54e-3` with a 95% importance-
  sampling interval `[1.51e-3, 1.56e-3]` at the 5 cm threshold.  They report
  A-bPCE at `1.49e-3` using 235 model calls.
- The conventional Timoshenko reconstruction reproduces the older benchmark
  moments of 0.069 ft and 0.021 ft.  The explicit Euler-Bernoulli sensitivity
  reproduces Li et al.'s direct-MCS moments of 0.0652 ft and 0.0202 ft.
- Wagner et al. (2022) use the same dependent input model at a 9 cm threshold
  and report a direct-MCS reference `Pf = 1.49e-6` from `10^8` evaluations.

The two moment pairs reflect an under-specified element formulation in the
published benchmark.  Both variants, the fixed modeling assumptions, and all
raw results are retained; no empirical response factor is used.

## Sources

- S. Marelli and B. Sudret, *Structural Safety* 75 (2018), 67-74,
  <https://doi.org/10.1016/j.strusafe.2018.06.003>.
- C. Li et al., *Engineering Computations* 35(6) (2018), 2165-2214,
  <https://doi.org/10.1108/EC-04-2017-0140>.
- P.-R. Wagner et al., *Structural Safety* 96 (2022), 102179,
  <https://doi.org/10.1016/j.strusafe.2021.102179>.
