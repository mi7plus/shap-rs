# shap-rs

Native Rust model explanations powered by Shapley values.

The crate supports exact interventional SHAP, reproducible permutation SHAP,
Kernel SHAP with constrained weighted least squares, closed-form linear SHAP, polynomial TreeSHAP,
and exact tree interaction values. Models can have
one or many outputs, and every explainer returns the same `Explanation` type.

## Highlights

- Batch-oriented model trait based on `ndarray`
- Background-distribution masking (not just a single mean replacement)
- Exact local accuracy for exact and linear explainers
- Deterministic sampling with configurable seeds and antithetic permutation pairs
- Additivity checks and plot-ready bar, force, waterfall, and beeswarm data
- No unsafe code

## Explainers

- Automatic Exact/Kernel selection for arbitrary prediction models
- Exact interventional SHAP
- Kernel SHAP with constrained weighted least squares
- Permutation and Sampling SHAP, including Monte-Carlo standard errors
- Independent and covariance-aware Linear SHAP
- Flat and hierarchical Partition SHAP/Owen values
- Polynomial TreeSHAP and polynomial-time exact tree interactions
- Expected Gradients/Gradient SHAP, including repeated-estimate uncertainty
- Framework-adapted Deep SHAP
- Asymmetric causal Shapley values
- Exact model-agnostic interaction values

All model-agnostic explainers accept custom `Masker` implementations. Built-in
maskers cover background-distribution replacement, fixed references, numeric
text tokens, flattened images, and closure-defined conditional sampling.
Any explainer can be wrapped with `ExplainerExt::with_metadata()` so validated
feature and output metadata is attached automatically to every result.

## Optional features

- `json-adapters`: JSON explanation serialization plus XGBoost and LightGBM import
- `parallel`: Rayon-backed parallel execution over sample batches
- `burn-adapter`: Burn 0.15 autodiff integration via `burn_adapter::BurnModel`

The Burn adapter is optional to preserve the lightweight default build. It
accepts a tensor forward closure and implements both `Predict` and
`DifferentiablePredict`, so it can drive `GradientExplainer` directly. Burn
0.15 is intentionally pinned because newer Burn releases or their dependencies require a newer Rust toolchain,
while `shap-rs` supports Rust 1.80.

## Visualization

Every plot module exposes serializable plot-ready data. The `plot::svg` module
also renders dependency-free, standalone SVG for global importance,
per-sample waterfalls and force plots, beeswarms, heatmaps, dependence
scatters, and decision paths:

```rust
use shap_rs::plot::svg::{global_bar, SvgOptions};
# use ndarray::{array, Array3};
# use shap_rs::Explanation;
# let explanation = Explanation::new(
#     Array3::from_shape_vec((1, 2, 1), vec![1.0, -0.5]).unwrap(),
#     array![[0.0]], array![[2.0, 3.0]])?;
let svg = global_bar(&explanation, &SvgOptions::default())?;
assert!(svg.starts_with("<svg"));
# Ok::<(), shap_rs::ShapError>(())
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for extension points and invariants and
[NUMERICS.md](NUMERICS.md) for comparison tolerances. [ROADMAP.md](ROADMAP.md)
is the living implementation and release tracker.
Local Criterion methodology and the TreeSHAP parallel baseline are recorded in
[BENCHMARKS.md](BENCHMARKS.md).
Causal assumptions and serialized attribution semantics are documented in
[CAUSAL.md](CAUSAL.md).
Burn gradient/deep operation and input contracts are documented in
[NEURAL.md](NEURAL.md).
The reviewed public surface and pre-1.0 compatibility policy are recorded in
[API_STABILITY.md](API_STABILITY.md).

## Native TreeSHAP

```rust
use ndarray::array;
use shap_rs::{Explainer, MissingBranch, Node, Tree, TreeEnsemble};
use shap_rs::explainers::TreeExplainer;

let tree = Tree::new(vec![
    Node::Split { feature: 0, threshold: 0.0, left: 1, right: 2,
        missing: MissingBranch::Left, cover: 10.0 },
    Node::Leaf { values: vec![1.0], cover: 4.0 },
    Node::Leaf { values: vec![5.0], cover: 6.0 },
], 0, 1)?;
let model = TreeEnsemble::new(vec![(tree, 1.0)], vec![0.0])?;
let explanation = TreeExplainer::new(&model).explain(array![[2.0]].view())?;
assert!((explanation.reconstructed()[[0, 0]] - 5.0).abs() < 1e-12);
# Ok::<(), shap_rs::ShapError>(())
```

Node `cover` values encode the training mass reaching each branch. They are
required for path-dependent TreeSHAP expectations when a split feature is
absent. `NaN` values follow the node's configured missing branch.

For XGBoost DART recursive dumps, use
`from_xgboost_json_with_tree_weights` with the full model JSON's `weight_drop`
array; ordinary boosted-tree dumps use `from_xgboost_json`. Per-sample raw base
margins are supported by `TreeEnsemble::predict_with_base_margin` and
`TreeExplainer::explain_with_base_margin` and replace, rather than add to, the
model's fixed base offset.

`from_xgboost_model_json` imports XGBoost's full columnar saved-model schema,
including `tree_info` output groups and DART `weight_drop`. It accepts explicit
raw-margin base values so objective-specific base-score transforms remain
unambiguous. Full-schema categorical XGBoost splits are currently rejected;
numerical `gbtree` and `dart` models are supported.

`TreeEnsemble::output_groups` exposes optional per-tree output metadata.
Full-model XGBoost imports populate it from `tree_info`; recursive XGBoost and
LightGBM dumps retain their documented inferred ordering.

Imported trees retain their split semantics: XGBoost numeric splits use strict
`<`, LightGBM numeric splits use `<=`, and LightGBM categorical splits use
integer category membership. LightGBM `NaN`, `Zero`, and `None` missing-value
policies are preserved. Native callers can construct these explicitly with
`Node::NumericalSplit`, `Node::CategoricalSplit`, `SplitComparison`, and
`MissingValuePolicy`; the legacy `Node::Split` remains a `<=`/NaN split.

`TreeExplainer` computes polynomial tree-path-dependent values from node
covers. `InterventionalTreeExplainer` instead integrates absent features over
an explicit `Background`; it is exact and currently limited to small feature
sets because it enumerates coalitions.

The interventional explainer also provides `explain_probability` (sigmoid for
one output, softmax for multiple outputs) and `explain_binary_log_loss` with
per-sample targets. These explain the transformed function exactly; raw-margin
output remains the default for both tree explainers.

Large hierarchical Owen explanations can opt into deterministic Monte Carlo
fallback with `HierarchicalPartitionExplainer::with_approximate_samples` when
the exact hierarchy permutation count exceeds `max_permutations`.

`ConditionalTabularMasker` selects nearest background rows using exact-match
distance for configured categorical columns and variance-scaled distance for
numeric columns. Conditioning uses only coalition-present features; with no
features present it retains the complete background distribution.

## Quickstart
```rust
use shap_rs::explain_sample;

let sample = vec![1.0, 2.0];
let background = vec![vec![0.0, 0.0]];
let predict_fn = |batch: &[Vec<f64>]| batch.iter().map(|x| x[0] + x[1]).collect();

let attributions = explain_sample(predict_fn, &sample, &background, 64).unwrap();
assert!((attributions.iter().sum::<f64>() - 3.0).abs() < 1e-9);
```

## Typed API

```rust
use ndarray::{array, Array2, ArrayView2};
use shap_rs::{Background, Explainer, FnModel};
use shap_rs::explainers::ExactExplainer;

let model = FnModel::new(|x: ArrayView2<'_, f64>| {
    let mut y = Array2::zeros((x.nrows(), 1));
    for i in 0..x.nrows() { y[[i, 0]] = 2.0 * x[[i, 0]] - x[[i, 1]]; }
    Ok(y)
});
let background = Background::new(array![[0.0, 0.0], [1.0, 1.0]])?;
let explanation = ExactExplainer::new(model, background).explain(array![[3.0, 2.0]].view())?;
# Ok::<(), shap_rs::ShapError>(())
```

`ExactExplainer` is exponential and defaults to a 20-feature safety limit.
`KernelExplainer` enumerates the Shapley-kernel design for small feature sets
and uses deterministic complement-paired coalition sampling for larger ones.
Its constrained weighted solve enforces local accuracy exactly.

Licensed under MIT.
