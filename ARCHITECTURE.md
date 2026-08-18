# shap-rs architecture

`shap-rs` separates model execution, missing-feature semantics, attribution
algorithms, and presentation. An explainer never owns assumptions hidden inside
a model adapter.

```text
Predict / DifferentiablePredict / DeepAttribution
                         |
                       Masker
                         |
       Background / partitions / causal graph
                         |
                      Explainer
                         |
                     Explanation
           /             |             \
    additivity       interactions    plot data
```

## Core invariants

- Model input and output are batched `ndarray` values.
- SHAP values use `(samples, features, outputs)` order.
- Interaction values use `(samples, feature, feature, outputs)` order.
- Base values use `(samples, outputs)` order.
- Exact explainers enforce local accuracy up to numerical tolerance.
- Approximate explainers expose deterministic seeds; Sampling SHAP can report
  standard errors.
- Model-agnostic coalition evaluation is deduplicated, bounded, and batched.
- Serialized explanations carry a schema version and are revalidated when read.
- Tree covers represent training mass and determine absent-branch probabilities.

## Extension points

Implement `Predict` for a new model runtime. Implement `Masker` when absent
features need conditional or structured replacement. Autodiff integrations use
`DifferentiablePredict`; neural graph Deep SHAP adapters use `DeepAttribution`.
Accelerated runtimes implement `AcceleratedPredict` and bind a device through
`DeviceModel`.

Tree frameworks should convert into `TreeArrays` or the native `Tree` model.
The `json-adapters` feature includes direct XGBoost and LightGBM dump readers.
Native-library adapters are added only when an upstream public API exposes the
complete topology, split comparison, leaf outputs, node covers, and missing
routing needed by TreeSHAP. SmartCore keeps its base-tree representation
crate-private, while Linfa Trees exposes traversal and splits but not covers or
missing routing. Prediction-only adapters would silently change expected-value
semantics, so both libraries should continue through caller-built `TreeArrays`
until those fields are public.

## Resource control

`EvaluationConfig` bounds coalition batch size, retained cache entries, and
optionally the number of model rows evaluated per explained sample. Exponential
algorithms additionally expose feature, permutation, or ordering limits.
