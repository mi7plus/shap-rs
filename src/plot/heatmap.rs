use crate::{Explanation, Result, ShapError};
use ndarray::Array2;
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct HeatmapData {
    pub values: Array2<f64>,
    pub feature_order: Vec<usize>,
}
pub fn data(e: &Explanation, output: usize) -> Result<HeatmapData> {
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    let order = crate::plot::bar::data(e)
        .into_iter()
        .map(|x| x.0)
        .collect::<Vec<_>>();
    let values = Array2::from_shape_fn((e.n_samples(), e.n_features()), |(i, j)| {
        e.values()[[i, order[j], output]]
    });
    Ok(HeatmapData {
        values,
        feature_order: order,
    })
}
