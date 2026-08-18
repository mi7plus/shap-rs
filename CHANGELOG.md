# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the crate follows
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Metadata-aware interaction explanations.
- Selection and concatenation for interaction and uncertain explanations.
- An optional Burn 0.15 autodiff adapter for prediction and expected gradients.
- Regenerable Python SHAP 0.46 compatibility fixtures for Exact, Kernel,
  Permutation, Partition, Linear, Tree, and tree-interaction explainers.
- Trained XGBoost and LightGBM regression, binary, and multiclass fixtures,
  including missing-value routing and raw-margin output ordering.
- Grouped feature masking with source-column mappings and group-axis explanation
  data across coalition explainers.
- A committed Burn affine fixture covering Expected Gradients and Deep
  attributions, base values, and reconstructed predictions.
- A documented pre-1.0 public API and SemVer policy.
- Metadata-aware SVG categorical ticks, missing-value legends, feature units,
  and probability/log-odds/class-score output labels.
- Correct LightGBM TreeSHAP cover selection for classification models by
  preferring node sample counts over Hessian weights.
- XGBoost-compatible per-sample base margins for native prediction and
  TreeSHAP, plus explicit DART `weight_drop` import support.
- Checked allocation-size guards across core explainer outputs, interaction
  tensors, gradient batches, coalition batches, and tree prediction buffers.
- Explicit strict/inclusive numerical and categorical tree split semantics,
  including LightGBM categorical and NaN/zero/none missing-value routing.
- Direct XGBoost full saved-model JSON import for numerical `gbtree` and DART
  models, using columnar arrays, `tree_info`, and `weight_drop` metadata.
- Exact background-distribution `InterventionalTreeExplainer`, distinct from
  polynomial tree-path-dependent `TreeExplainer` semantics.
- Exact interventional probability and binary logistic-loss tree explanations.
- Serialized per-tree output-group metadata, populated from XGBoost full-model
  `tree_info` rather than inferred multiclass ordering.
- Seeded hierarchy-consistent Monte Carlo fallback for large Owen hierarchies.
- Empirical nearest-neighbor conditional tabular masker with categorical and
  variance-scaled numerical distance.
- Sequential and Rayon TreeSHAP batch benchmarks with recorded methodology and
  a measured local parallel baseline.
- Serialized `AttributionSemantics`, with causal results explicitly marked
  asymmetric and causal assumptions documented.
- Explicit interventional, observational, and conditional tabular causal
  masking modes.
- Concrete affine Burn `DeepAttribution` adapter with documented operation,
  input, state, and structured-output boundaries.
- Tokenizer-aware text masking with piece reconstruction, special-token
  policies, and grouped subword coalitions.
- Segment/superpixel image masking with fixed and blur baselines plus an
  inpainting callback adapter.
- Incremental out-of-core masked-background evaluation.
- Bounded row-level `CachedModel` prediction reuse across samples and explainer
  calls.
- A condition-preserving Householder QR backend for Kernel SHAP weighted least
  squares.
- Executable CPU/device numerical-equivalence and repeated-run determinism
  checks.
- Owned coalition-batch prediction paths for single-transfer accelerated model
  adapters.
- Native validated CSR matrices, sparse prediction and masking contracts, and
  sparse permutation SHAP without dense coalition materialization.

## [0.1.0] - 2026-08-17

### Added

- Initial native Rust implementation of model-agnostic, linear, partition,
  gradient, deep-adapter, causal, TreeSHAP, interaction, and plotting APIs.

[Unreleased]: https://github.com/mi7plus/shap-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mi7plus/shap-rs/releases/tag/v0.1.0
