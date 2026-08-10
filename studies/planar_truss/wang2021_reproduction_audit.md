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
- replacing the published score by the standard U-score at 55 calls.  Across
  all 13 development seeds, the mean estimate was 1.3278081e-3: its relative
  error was 11.96%, while the mean and median absolute per-run errors were
  13.45% and 9.96%.  The U-score is useful as a diagnostic baseline but is not
  competitive on this benchmark;
- estimating Pf from a weighted local affine approximation of the learned
  boundary.  The configuration fixed before extending the experiment
  (weight power 1 and ridge penalty 0.1) gave a 0.97% error of the mean across
  13 seeds, but a 17.68% mean absolute per-run error, a 14.59% median error,
  and a 44.61% worst-case error.  Leave-one-out aggregation did not reduce
  this dispersion.  A post-hoc grid contained settings with a nearly unbiased
  mean only because large positive and negative run errors cancelled; those
  settings are explicitly ineligible for confirmation or publication.
- multiplying the Gaussian sign-misclassification probability by the
  nearest-design distance.  This parameter-free boundary-diversity score
  improved substantially over the U-score on all 13 development seeds: the
  error of the mean fell from 11.96% to 3.35%, and the mean and median
  absolute per-run errors fell from 13.45% and 9.96% to 8.64% and 6.73%.
  Sensitivity rose from 0.846 to 0.874.  However, the across-seed standard
  deviation increased slightly to 1.939e-4 and the worst run still had a
  35.31% error.  It is therefore an informative acquisition ablation, not a
  confirmation candidate;
- selecting the final model by sign-stratified cross-validation on the same
  13 boundary-diversity designs.  Its mean happened to have 0.23% relative
  error, but the mean absolute per-run error was 8.36% and the worst run was
  42.08%.  In that worst run, the out-of-fold balanced sign error was zero,
  demonstrating that sign accuracy on an adaptively concentrated design is
  not a reliable proxy for the probability mass of the global failure domain.
- using the fixed C = 1e3, epsilon = 1e-3, theta = 0.00625 profile throughout
  a boundary-diversity campaign.  The three-seed gate reduced the standard
  deviation to 2.382e-5, but all three runs overestimated failure probability;
  the error of the mean and the mean absolute error were both 4.71%, compared
  with 2.35% and 2.75% for evidence tuning on the same seeds.  The remaining
  ten trajectories were therefore not run;
- multiplying boundary diversity by the normalized gradient factor with unit
  weight.  Its paired three-seed errors were 4.65%, 8.00%, and 19.57%, versus
  6.73%, 0.60%, and 0.92% without the gradient.  The large deterioration in
  the third run failed the extension gate.
- replacing the log-ratio target by the conventional dimensional displacement
  margin.  Its paired three-seed errors were 22.67%, 11.22%, and 19.63%, all
  much worse than the log-ratio campaign.  The equivalent log response is
  therefore retained as a deliberate regression-conditioning choice, without
  claiming it reproduces the response used by Wang et al.

The last result is stored in
`results/planar_truss/pool20_development_seed11.json`.  None of the discarded
variants may be selected post hoc for the article.

## Next controlled comparison

The acquisition experiments indicate that model selection, rather than lack of
candidate diversity alone, is the next bottleneck.  The current deterministic
evidence grid repeatedly changes an isotropic model during enrichment, while
sign-stratified cross-validation is overconfident on the concentrated adaptive
design.

A small global grid, evaluated on all 13 development designs, identified a
stable region at C = 1e3, epsilon = 1e-3, and theta = 0.00625.  Using this one
profile only for final refitting reduced the across-seed standard deviation to
1.073e-4, the mean absolute error to 5.53%, and the worst error to 15.02%; its
mean still had 1.54% error.  The adaptive test above showed that this stability
does not carry over when the same profile controls enrichment.

The next controlled diagnostic keeps the better log-ratio response and revisits
anisotropic optimization without the unstable all-at-once box-min search.  On
each already evaluated design, it will start from the selected isotropic
evidence model, hold C and epsilon fixed, and perform one deterministic
coordinate sweep over each theta_k using multiplicative factors
{0.25, 0.5, 1, 2, 4}.  The procedure uses training evidence only and costs no
new limit-state calls.  A three-seed refit gate will determine whether it is
worth embedding the refinement inside the adaptive loop.

The squared-epsilon branch already implements the exact active-set curvature:
its dual contains the separate 1/C multiplier penalties, its predictive
covariance uses the active Hessian C I, and its evidence determinant is
consistent with that same loss.  Thus the thesis reviewers' concern about the
heuristic H approximately equal to lambda I is already addressed; merely
renaming this implementation as an exact-Hessian modification would not be a
new contribution.
