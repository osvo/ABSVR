# Wang et al. (2021) reproduction and development audit

## Purpose

This note prevents the structural benchmark from being presented as a faithful
reproduction when the implemented protocol differs from the published ABSVR
algorithm.  It also records the development/confirmation split used while
searching for a competitive, reference-blind modification.

The primary metrics are the relative error of the mean failure-probability
estimate against the independent RQMC reference and the number of true
limit-state calls.  Wall-clock time is deliberately excluded.  Per-run mean,
median, range, and standard deviation remain mandatory robustness diagnostics.

## Double-checked literature facts

The following items were checked against both the institutional accepted
manuscript and the publisher article:

- Wang et al. define ABSVR1 with the square loss and ABSVR2 with the
  epsilon-insensitive square loss.  Consequently, the repository's
  `squared_epsilon` option is an ABSVR2-type loss and is not, by itself, a new
  contribution.
- The initial design contains 15 Latin-hypercube points.  The candidate set is
  generated with a Sobol sequence.
- The published initial hyperparameters are eta = 1e5, epsilon = 1e-5, and one
  theta parameter per input initialized to 1.  The reported optimization
  bounds are [10, 1e10], [1e-8, 0.01], and [1e-5, 1e5], respectively.
- The adaptive sampling-region threshold is the empirical
  `0.1 * estimated_Pf` quantile of the joint input density.
- The published learning score combines proximity to the predicted limit
  state, predictive standard deviation, joint density, and nearest-design
  distance.
- The stopping rule is hybrid: stability of the reliability index and a
  bootstrap-confidence error bound.  The default thresholds are 1e-4, 0.1,
  0.01, and 1e-2.  The candidate population must also give an estimated
  coefficient of variation below 5%.
- The truss results are averages of ten runs: ABSVR1 reports Pf = 1.538e-3 and
  57.8 calls; ABSVR2 reports Pf = 1.512e-3 and 54.9 calls.  Table 8 reports
  1.18% and 0.53% errors against its 1.52e-3 MCS value.  Against this
  repository's more precise independent RQMC mean (1.5082359313964844e-3), the
  corresponding errors are approximately 1.973% and 0.250%.

Sources:

- Wang, Li, Xu, Li, and Kareem, *Computer Methods in Applied Mechanics and
  Engineering* 387 (2021), 114172,
  <https://doi.org/10.1016/j.cma.2021.114172>.
- Swansea University accepted-manuscript record,
  <https://cronfa.swansea.ac.uk/Record/cronfa58159>.

## Material deviations in the current campaign

The completed `confirmation_campaign_v2.json` is a valid evaluation of the
repository algorithm, but not an exact reproduction of published ABSVR2:

1. It forces 80 enrichment points (95 total calls) instead of applying the
   published hybrid stopping rule.
2. It selects one isotropic theta from a deterministic evidence grid and
   retunes every 20 points.  Wang et al. optimize anisotropic theta values with
   continuous bounded optimization during refinement.
3. It works in independent standard-normal coordinates, whereas Wang et al.
   explicitly formulate the algorithm without an isoprobabilistic
   transformation.
4. It uses a fixed 2^17 candidate population.  At Pf about 1.5e-3, its nominal
   binomial CoV is about 7.1%, above the published 5% requirement.
5. It uses the equivalent log-ratio response rather than the dimensional
   displacement-margin response.  The failure event is unchanged, but the
   regression problem is not identical.

These differences must be described as deliberate implementation choices or
removed before making a reproduction claim.

## Competitive target

The preferred target is a confirmed mean call count no greater than 54.9 and a
relative error of the mean no greater than 0.250% against the independent RQMC
reference.  A result outside that rectangle is not automatically invalid, but
it must establish a clear, uncertainty-aware Pareto improvement rather than
relying on a favorable single run.

## Development and confirmation firewall

All seeds whose results have already been inspected against the reference are
development seeds and cannot be reused for final confirmation:

`11, 53, 101, 127, 139, 151, 163, 179, 191, 211, 223, 239, 251`.

The final algorithm, call rule, estimator, and every tuning constant will be
frozen before running new, disjoint confirmation seeds.  Reference probability
is allowed only to rank development variants and to audit the frozen
confirmation result; it is never an algorithm input.

## Discarded development hypotheses

The following changes were rejected because they did not improve both primary
metrics consistently:

- continuous anisotropic box-min evidence optimization initialized exactly as
  reported by Wang et al.; it converged to highly local models that predicted
  almost no failures on the three development designs;
- robust intercept calibration from five-fold out-of-fold boundary residuals;
- a linear ridge trend followed by an SVR residual model;
- standalone regularized degree-one and degree-two polynomial trends at 55
  calls;
- training the stored designs in standardized physical coordinates;
- evidence-weighted averaging over several SVR hyperparameter models;
- replacing hard classification with the posterior mean
  `mean(Phi(-mu/sigma))`; the favorable three-seed result at 45 calls did not
  replicate over ten additional development trajectories;
- increasing the candidate population to 2^20 while retaining the remaining
  protocol.  The first development run used 55 true calls and produced
  Pf = 8.4114075e-4, a 44.23% relative error, so the campaign was stopped.

The last result is stored in
`results/planar_truss/pool20_development_seed11.json`.  None of the discarded
variants may be selected post hoc for the article.

## Next controlled comparison

The next comparison will change only the acquisition score while retaining the
same initial-design generator, squared-epsilon BSVR, 2^17 candidate set, and
55-call budget.  A standard misclassification/U score will be tested first as
a diagnostic baseline.  Any proposed new score must be defined before looking
at its reference error, evaluated on all development seeds, and then frozen for
new-seed confirmation.
