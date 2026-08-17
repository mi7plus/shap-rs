use crate::{Explanation, Result, ShapError};
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ForceData {
    pub base_value: f64,
    pub output_value: f64,
    pub contributions: Vec<f64>,
}
pub fn data(e: &Explanation, sample: usize, output: usize) -> Result<ForceData> {
    if sample >= e.n_samples() {
        return Err(ShapError::InvalidSampleIndex {
            index: sample,
            n_samples: e.n_samples(),
        });
    }
    if output >= e.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: e.n_outputs(),
        });
    }
    let contributions = (0..e.n_features())
        .map(|j| e.values()[[sample, j, output]])
        .collect::<Vec<_>>();
    Ok(ForceData {
        base_value: e.base_values()[[sample, output]],
        output_value: e.base_values()[[sample, output]] + contributions.iter().sum::<f64>(),
        contributions,
    })
}
