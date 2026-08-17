use crate::{Explanation, Result, ShapError};
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScatterPoint {
    pub sample: usize,
    pub feature_value: f64,
    pub shap_value: f64,
    pub color_value: Option<f64>,
}
pub fn data(
    e: &Explanation,
    feature: usize,
    output: usize,
    color_feature: Option<usize>,
) -> Result<Vec<ScatterPoint>> {
    if feature >= e.n_features() {
        return Err(ShapError::InvalidFeatureIndex {
            index: feature,
            n_features: e.n_features(),
        });
    }
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    if let Some(index) = color_feature.filter(|&j| j >= e.n_features()) {
        return Err(ShapError::InvalidFeatureIndex {
            index,
            n_features: e.n_features(),
        });
    }
    Ok((0..e.n_samples())
        .map(|i| ScatterPoint {
            sample: i,
            feature_value: e.data()[[i, feature]],
            shap_value: e.values()[[i, feature, output]],
            color_value: color_feature.map(|j| e.data()[[i, j]]),
        })
        .collect())
}
