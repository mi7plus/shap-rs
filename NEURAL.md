# Neural adapter contract

The optional `burn-adapter` feature exposes two deliberately different paths:

- `BurnModel` adapts a caller-supplied single dense tensor graph to prediction
  and autodiff gradients. It supports any Burn operations whose backward pass
  retains the input gradient and is intended for `GradientExplainer`.
- `BurnAffineModel` is the concrete `DeepAttribution` adapter. Its supported
  graph is exactly one affine operation, `x.matmul(weights) + bias`; its Deep
  SHAP contributions are exact relative to the mean background.

Unsupported nonlinear Deep SHAP operations return no fallback because
`BurnAffineModel` cannot represent them. Use `BurnModel` with expected
gradients, or implement `DeepAttribution` for a graph whose operation rules are
known.

Both built-in adapters accept one rank-2 dense input `(samples, features)` and
produce one rank-2 dense output `(samples, outputs)`. Multiple inputs,
embeddings, recurrent/hidden state, ragged tensors, and structured outputs are
not flattened implicitly. Callers must expose an explicit dense model boundary
or provide a custom adapter. Device behavior is determined by the supplied Burn
backend/device; conversion to the public explanation format uses `f64`.
