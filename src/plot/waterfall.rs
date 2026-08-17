use crate::{Explanation, Result, ShapError};
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct WaterfallItem {
    pub feature: usize,
    pub value: f64,
    pub contribution: f64,
}
pub fn data(e: &Explanation, sample: usize, output: usize) -> Result<Vec<WaterfallItem>> {
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
    let mut r = (0..e.n_features())
        .map(|j| WaterfallItem {
            feature: j,
            value: e.data()[[sample, j]],
            contribution: e.values()[[sample, j, output]],
        })
        .collect::<Vec<_>>();
    r.sort_by(|a, b| b.contribution.abs().total_cmp(&a.contribution.abs()));
    Ok(r)
}
