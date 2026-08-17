use crate::{Explainer, Explanation, Result};
use ndarray::{Array2, Array3, ArrayView2};
/// Explainer for generalized additive models. The callback returns base values
/// and already-separated per-feature term contributions.
pub struct AdditiveExplainer<F> {
    decompose: F,
}
impl<F> AdditiveExplainer<F> {
    pub fn new(decompose: F) -> Self {
        Self { decompose }
    }
}
impl<F> Explainer for AdditiveExplainer<F>
where
    F: Fn(ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array3<f64>)>,
{
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let (base, values) = (self.decompose)(x)?;
        Explanation::new(values, base, x.to_owned())
    }
}
