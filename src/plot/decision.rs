use crate::{Explanation, Result, ShapError};
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DecisionPath {
    pub sample: usize,
    pub feature_order: Vec<usize>,
    pub cumulative_values: Vec<f64>,
}
pub fn data(e: &Explanation, output: usize) -> Result<Vec<DecisionPath>> {
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
    Ok((0..e.n_samples())
        .map(|i| {
            let mut cumulative = vec![e.base_values()[[i, output]]];
            for &j in &order {
                cumulative.push(cumulative.last().copied().unwrap() + e.values()[[i, j, output]])
            }
            DecisionPath {
                sample: i,
                feature_order: order.clone(),
                cumulative_values: cumulative,
            }
        })
        .collect())
}
