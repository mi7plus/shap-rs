# Numerical tolerances

Floating-point comparisons use combined absolute and relative tolerances:

```text
|actual - expected| <= absolute + relative * max(|actual|, |expected|)
```

The following defaults are recommended for `f64` results. Reference fixtures
may loosen them only with an explanation recorded beside the fixture.

| Algorithm | Absolute | Relative | Notes |
| --- | ---: | ---: | --- |
| Exact SHAP | `1e-10` | `1e-9` | Exhaustive coalition summation |
| Linear SHAP, independent | `1e-11` | `1e-10` | Closed form |
| Linear SHAP, correlated | `1e-8` | `1e-7` | Covariance solve and regularization |
| Native TreeSHAP | `1e-10` | `1e-9` | Path-dependent raw margin |
| Exact interactions | `1e-10` | `1e-9` | Also used for symmetry checks |
| Kernel SHAP | `1e-7` | `1e-6` | Weighted least-squares conditioning |
| Permutation/Partition sampling | `1e-6` | `1e-5` | Deterministic algorithm checks only |
| Expected Gradients | `1e-5` | `1e-4` | Deterministic linear reference cases |
| SVG plot coordinates | `1e-6` | `1e-6` | Compare parsed numeric attributes |

Monte-Carlo explainers should primarily be checked statistically using their
reported standard errors. A reference result is accepted when each difference
is within the larger of the table tolerance and three combined standard errors.
Tests of reproducibility compare values exactly when seed, platform, feature
set, and execution mode are identical.

Cross-platform fixtures use the listed tolerances on Linux, Windows, and macOS.
Tests must not depend on fused multiply-add behavior or iteration order from an
unordered collection. CPU and future GPU adapters must document any wider
backend-specific tolerance before being enabled in compatibility CI.

## Accelerated-device contract

`check_device_equivalence` makes the CPU/device contract executable. CPU
repeated runs are bit-exact by default. CUDA, Metal, Vulkan, and WebGPU use
combined `1e-5` absolute/relative CPU-equivalence tolerances and `1e-6`
repeated-run tolerances over three device executions. Models that require wider
tolerances must opt in explicitly and record the reason beside their fixture.
Non-finite values, changing output shapes, and drift outside the configured
limits fail validation. Determinism refers to fixed inputs, model state, device,
and build; stochastic models must freeze their RNG before validation.

Kernel SHAP defaults to its allocation-light normal-equations solver. For
large or poorly conditioned sampled designs, select
`KernelSolver::HouseholderQr`; it operates on the weighted design directly and
therefore avoids squaring its condition number. The QR backend is implemented
in-crate to preserve the Rust 1.80 MSRV and avoid a heavy mandatory linear
algebra dependency. Ridge rows are appended to the QR design exactly rather
than added after factorization.
