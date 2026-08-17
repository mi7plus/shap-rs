use crate::Explanation;
use ndarray::{Array1, Axis};
/// Global feature importance: mean absolute SHAP value over samples and outputs.
pub fn mean_absolute_shap(e: &Explanation) -> Array1<f64> {
    let mut out = Array1::zeros(e.n_features());
    for j in 0..e.n_features() {
        out[j] = e
            .values()
            .index_axis(Axis(1), j)
            .iter()
            .map(|x| x.abs())
            .sum::<f64>()
            / (e.n_samples() * e.n_outputs()) as f64
    }
    out
}
