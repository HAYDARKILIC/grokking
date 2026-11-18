# Reproducing the study

## 1. Smoke test first

Open `notebooks/grokking_wd_clock.ipynb`, leave `SMOKE_TEST = True`, and run all
cells. This exercises the dataset, the model, both optimizers, the projector,
all four phases, the figures, the tables and the manifest in a few minutes. If
it completes, the pipeline is intact.

## 2. Full grid

Set `SMOKE_TEST = False` and run all cells again. Expect roughly 12 accelerator
hours for 351 runs on an A100. Records accumulate in `runs/runs.jsonl`; the
runner skips configurations already present, so the session can be interrupted
and resumed.

## 3. What each phase should produce

**Phase A.** All four landmarks within a factor of two of the reported values.
`k_mem` is expected to come out early (we measured 225 against a reported 400);
the delay ratio inherits that and lands near 40 against a reported 26. This is
within the seed distribution that Phase B measures, and is not a failure.

**Phase B.** Complete success at p = 17 and p = 23, and **no** grokking at
p = 31 within the 400,000-iteration ceiling. The delay ratio should increase
approximately linearly with width and comparably with the initialization scale.
Expect the fitted parameter-norm rate to be negative in about two thirds of
grokked runs at p = 23: the norm grows while the network generalizes.

**Phase C.** The β = 0 arm is an algebraic null control and must return a ratio
of 1 within its scatter; that scatter is the resolution of the phase (we
measured ±15 %). Agreement at β = 0.5, and an excess of 36 % and 17 % at
β = 0.9 and 0.95.

**Phase D.** The training-Jacobian leakage must sit at the double-precision
floor (we measured ≤ 4.86e-12); if it does not, the projector is wrong and
nothing downstream is interpretable. Drift at c = 1 must be at machine epsilon.
The frozen component should be non-monotone in every run.

## 4. Interpreting the (P2) result

The slope agreeing with `log(c)/r` is close to arithmetically forced, given an
exponential mode and a threshold fixed at a multiple of the unrescaled
amplitude. The substantive quantity is the **rate invariance**: the fitted
component rate varies by 0.247 % across the eleven rescalings. A nonlinear
coupling would have changed it. The notebook prints this number explicitly.

## 5. Additional controls

Set `RUN_ADDITIONAL_CONTROLS = True` in the final cell to run:

- **C2**, Phase C at s ∈ {1.0, 2.5} — removes the conditionality on the
  calibrated initialization scale (~3.1 GPU-h).
- **C3**, Phase C with the component tracked — lets the rate ratio be formed on
  the theory's own observable rather than on the parameter norm (~1.6 GPU-h).
  One projector is built per (β, seed) on the coupled arm and reused for both
  arms, so the ratio stays a like-for-like comparison.
- **B31**, p = 31 at twice the ceiling — resolves the right-censoring (~9.6 GPU-h).

Run these in the **same numerical environment** as the main grid.

## 6. Numerical environment

Binary64 throughout, TF32 disabled at both the matrix-multiply and convolution
levels. This is not a stylistic choice: the projector is asserted at the 1e-14
level and float32 rounding alone leaves a residual near 1e-7, which would swamp
the intervention test.

Deterministic kernels are requested with warn-only semantics, so bitwise
reproducibility across executions is not guaranteed. Every reported quantity is
an average over thousands of iterations or a rate fitted over hundreds of
samples, so the residual nondeterminism is immaterial at the precision at which
conclusions are drawn — but CPU-versus-GPU execution is a different matter and
is known to flip individual seed outcomes. Pin the environment and record it.
