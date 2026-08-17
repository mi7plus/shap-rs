use crate::Explanation;
/// Feature indices and mean absolute SHAP values, largest first.
pub fn data(e: &Explanation) -> Vec<(usize, f64)> {
    let v = crate::metrics::mean_absolute_shap(e);
    let mut r = v.iter().copied().enumerate().collect::<Vec<_>>();
    r.sort_by(|a, b| b.1.total_cmp(&a.1));
    r
}
