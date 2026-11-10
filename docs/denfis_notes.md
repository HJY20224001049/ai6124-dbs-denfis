# DENFIS implementation notes

Status: implementation and experiments in progress, 2026-11-10. This is a
DENFIS-inspired implementation, not an exact reproduction of every 2002 option.

## Sources and scope

[Kasabov and Song (2002)](https://doi.org/10.1109/91.995117) motivate evolving
local fuzzy models. The ECM geometry follows
[Song and Kasabov's original ECM paper](https://kedri.aut.ac.nz/__data/assets/pdf_file/0018/23256/qs2001_36.pdf).
The project fixes Gaussian antecedents and local weighted RLS; these engineering
choices and the bounded-memory extension must be distinguished from the source.

## Notation and equations

For standardised input x in R^d, centre c_i and radius r_i, use normalised
Euclidean distance D_i = ||x-c_i|| / sqrt(d). ECM leaves a covering cluster's
geometry unchanged. Otherwise choose i minimising S_i = D_i+r_i. Create a rule
when S_i > 2 Dthr, unless the 200-rule cap is reached. Otherwise set
r_i' = S_i/2 and c_i' = c_i + (D_i-r_i')/D_i * (x-c_i).
At the cap, retain geometry and assign support to the closest rule.

Set sigma_i = max(r_i, 0.1 Dthr, 1e-6). Log firing is
l_i = -mean_j((x_j-c_ij)^2)/(2 sigma_i^2); normalise by softmax(l).
Equivalently, per-coordinate Gaussians have width sqrt(d)*sigma_i.
For phi = [1,x], local output is f_i = theta_i^T phi and the ensemble output
is sum_i h_i f_i. Compute softmax after subtracting its maximum.

Each active rule minimises a local weighted regression loss, not the global
ensemble loss. With v=P_i phi:

- K_i = h_i v / (lambda + h_i phi^T v).
- theta_i <- theta_i + K_i (y - theta_i^T phi).
- P_i <- (P_i - K_i v^T)/lambda.

Initial theta is zero; initial P is 100 I. Activations below 1e-6 are skipped,
including forgetting. Symmetrise P; reset that rule's P and coefficients if
nonfinite, nonpositive diagonal, or trace > 1e8. Count resets. Forecasts are
clipped to +/-0.2 log-return units; this fixed safeguard is not selected using
test outcomes. Synthetic regression may explicitly disable clipping.

## Training and online pseudocode

1. Reset every learned array and diagnostic counter.
2. Process training X chronologically with ECM; merge close centres.
3. Freeze antecedents and replay training (X,y) once through local RLS.
4. Frozen inference computes Gaussian weights and TSK outputs without mutation.
5. Online evaluation predicts first; after label realisation, partial_fit may
   update ECM/merge, then updates consequents. evolve_rules=False permits
   consequent-only adaptation; the frozen runner does not call partial_fit.

The two-pass fit uses only its training window. It is not a one-pass joint
antecedent/consequent training reproduction. No ECMc optimisation, original
triangular sets, top-m local selection, or higher-order offline DENFIS is claimed.

## Rule merging and complexity

When centre distance < merge_factor * Dthr, retain the higher-support centre
and consequent; enlarge its radius to enclose the removed cluster, then delete
the weaker rule. Radius may therefore exceed Dthr after a merge. Existing
consequents are not averaged. New online rules start with zero coefficients
and P=100 I. Repeated minimum-pair merging stops when no eligible pair remains.

For N examples, K<=200 rules and d features, fitting costs O(N K d^2),
prediction O(N K d), covariance storage O(K d^2). Pairwise merging requires
O(K^2 d) per scan. Online updates first screen the moved/new centre in O(K d);
unchanged geometry cannot introduce a new eligible pair. A differential test
checks equality with unconditional full scans. No random shuffle is
used, so the seed interface is deterministic without drawing random numbers.

## Evidence and limitations

Contract tests cover nonlinear held-out synthetic learning, reset/repeatability,
read-only inference, exact 1-D ECM geometry, the cap, forgetting, merge and
online growth, zero inputs, invalid shapes and nonfinite targets.
A first smoke run used the host Python 3.13 environment; formal artifacts are
rerun with Python 3.11 and pinned numerical dependencies. PyTorch/XGBoost are
not needed to run DENFIS; historical baseline artifacts are retained.
No performance gain, originality of DENFIS itself, or bonus grade is assumed.
