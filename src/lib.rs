#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
pub mod background;
pub mod coalition;
pub mod error;
pub mod evaluation;
pub mod explainer;
pub mod explainers;
pub mod explanation;
pub mod interactions;
pub mod link;
pub mod masker;
pub mod metadata;
pub mod metrics;
pub mod model;
#[cfg(feature = "parallel")]
pub mod parallel;
pub mod plot;
pub mod tree;
pub use background::Background;
pub use error::{Result, ShapError};
pub use evaluation::EvaluationConfig;
pub use explainer::{Explainer, ExplainerExt, MetadataExplainer};
pub use explanation::{Explanation, UncertainExplanation};
pub use link::Link;
pub use masker::{FixedMasker, FnMasker, ImageMasker, IndependentMasker, Masker, TextMasker};
pub use metadata::{FeatureKind, FeatureMetadata, OutputKind, OutputMetadata};
pub use model::{
    AcceleratedPredict, DeepAttribution, DeviceModel, DifferentiablePredict, ExecutionDevice,
    FnModel, Predict,
};
#[cfg(feature = "parallel")]
pub use parallel::ParallelExplainerExt;
pub use tree::{MissingBranch, Node, Tree, TreeArrays, TreeEnsemble};

/// Compatibility helper for scalar-output Kernel SHAP.
pub fn explain_sample<F>(
    predict: F,
    sample: &[f64],
    background: &[Vec<f64>],
    nsamples: usize,
) -> Result<Vec<f64>>
where
    F: Fn(&[Vec<f64>]) -> Vec<f64>,
{
    use ndarray::{Array2, ArrayView2};
    let m = sample.len();
    if background.is_empty() {
        return Err(ShapError::EmptyBackground);
    }
    if background.iter().any(|r| r.len() != m) {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{m} features"),
            found: "ragged background".into(),
        });
    }
    let bg = Background::new(
        Array2::from_shape_vec(
            (background.len(), m),
            background.iter().flatten().copied().collect(),
        )
        .map_err(|e| ShapError::Other(e.to_string()))?,
    )?;
    let model = FnModel::new(move |x: ArrayView2<'_, f64>| {
        let rows = x.rows().into_iter().map(|r| r.to_vec()).collect::<Vec<_>>();
        let y = predict(&rows);
        if y.len() != x.nrows() {
            return Err(ShapError::OutputDimensionMismatch {
                expected: x.nrows(),
                found: y.len(),
            });
        }
        Array2::from_shape_vec((y.len(), 1), y).map_err(|e| ShapError::ModelError(e.to_string()))
    });
    let e = explainers::KernelExplainer::new(model, bg)
        .with_nsamples(nsamples)
        .explain(
            Array2::from_shape_vec((1, m), sample.to_vec())
                .unwrap()
                .view(),
        )?;
    Ok((0..m).map(|j| e.values()[[0, j, 0]]).collect())
}
