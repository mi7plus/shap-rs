use crate::{Explanation, Result, ShapError};
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BeeswarmPoint {
    pub sample: usize,
    pub feature: usize,
    pub feature_value: f64,
    pub shap_value: f64,
}
pub fn data(e: &Explanation, output: usize) -> Result<Vec<BeeswarmPoint>> {
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    let order = crate::plot::bar::data(e);
    let mut r = Vec::with_capacity(e.n_samples() * e.n_features());
    for (feature, _) in order {
        for sample in 0..e.n_samples() {
            r.push(BeeswarmPoint {
                sample,
                feature,
                feature_value: e.data()[[sample, feature]],
                shap_value: e.values()[[sample, feature, output]],
            })
        }
    }
    Ok(r)
}
