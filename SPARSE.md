# Sparse model support

`SparseMatrix` is a validated, dependency-free CSR boundary. Rows require
sorted unique column indices and omit explicit zeros. `SparsePredict` and
`FnSparseModel` let a model consume this representation directly.

`SparsePermutationExplainer` uses `SparseIndependentMasker` and preserves CSR
storage through background replacement, coalition batching, caching, and model
prediction. It never creates a dense coalition matrix. The returned public
`Explanation` still contains dense display data, so the original input is
converted once after attribution; SHAP values are dense by definition because
every feature receives an attribution.

```rust
use ndarray::{array, Array2};
use shap_rs::{FnSparseModel, SparseMatrix, SparsePermutationExplainer};

let background = SparseMatrix::from_dense(array![[0.0, 0.0]].view())?;
let input = SparseMatrix::from_dense(array![[2.0, 3.0]].view())?;
let model = FnSparseModel::new(|x: &SparseMatrix| {
    Ok(Array2::from_shape_fn((x.nrows(), 1), |(row, _)| {
        x.row(row).unwrap().values().iter().sum()
    }))
});
let explanation = SparsePermutationExplainer::new(model, background)?
    .with_n_permutations(32)
    .explain(&input)?;
# Ok::<(), shap_rs::ShapError>(())
```

The sparse permutation path currently supports at most 62 coalition features,
matching the `u64` coalition representation used by dense model-agnostic
explainers. Missing values may be stored explicitly as `NaN`; model adapters
remain responsible for interpreting them.
