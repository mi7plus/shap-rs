# Causal and asymmetric attribution semantics

`CausalExplainer` restricts feature arrival orders to topological orders of a
caller-supplied directed acyclic graph. Its values answer an order-constrained
allocation question; they are not ordinary interventional SHAP values and need
not be symmetric when causal order is reversed.

The graph expresses permitted ordering, not a learned causal model. Distribution
semantics come from the supplied masker:

- `IndependentMasker` gives marginal background replacement.
- `ConditionalTabularMasker` gives an empirical nearest-neighbor conditional
  approximation based on coalition-present features.
- `FnMasker` can encode a domain-specific observational, interventional, or
  structural-causal sampler.

These choices require assumptions that the graph is appropriate, background
rows represent the target population, and the masker approximates the intended
intervention or observation. The crate does not identify causal effects from
data and does not validate absence of unmeasured confounding.

`CausalTabularMasker` makes these choices explicit through `interventional`,
`observational`, and `conditional` constructors. Observational and conditional
modes currently share the empirical nearest-neighbor sampler but retain a
distinct `CausalMaskingMode` so callers and downstream systems can preserve the
intended interpretation.

Every causal result is serialized with
`AttributionSemantics::CausalAsymmetric`. Other explicit semantics include
`Interventional`, `Conditional`, and `TreePathDependent`; legacy or custom
results default to `Unspecified`.
