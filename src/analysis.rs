//! Statistical aggregation of local explanations into global summaries.

use crate::{Explanation, Result, ShapError};
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuantileSummary {
    pub probability: f64,
    pub values: Array2<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttributionSummary {
    pub sample_count: usize,
    pub mean: Array2<f64>,
    pub standard_deviation: Array2<f64>,
    pub mean_absolute: Array2<f64>,
    pub quantiles: Vec<QuantileSummary>,
}

/// Summarizes every sample in an explanation.
pub fn summarize(explanation: &Explanation, quantiles: &[f64]) -> Result<AttributionSummary> {
    let indices = (0..explanation.n_samples()).collect::<Vec<_>>();
    summarize_samples(explanation, &indices, quantiles)
}

/// Summarizes a checked sample selection.
pub fn summarize_samples(
    explanation: &Explanation,
    samples: &[usize],
    quantiles: &[f64],
) -> Result<AttributionSummary> {
    explanation.validate()?;
    if samples.is_empty() {
        return Err(ShapError::EmptyData);
    }
    if let Some(&index) = samples.iter().find(|&&i| i >= explanation.n_samples()) {
        return Err(ShapError::InvalidSampleIndex {
            index,
            n_samples: explanation.n_samples(),
        });
    }
    validate_quantiles(quantiles)?;
    let features = explanation.n_features();
    let outputs = explanation.n_outputs();
    crate::error::checked_f64_shape(&[features, outputs], "attribution summary")?;
    crate::error::checked_f64_shape(
        &[quantiles.len(), features, outputs],
        "attribution quantiles",
    )?;
    let count = samples.len() as f64;
    let mut mean = Array2::zeros((features, outputs));
    let mut mean_absolute = Array2::zeros((features, outputs));
    for &sample in samples {
        for feature in 0..features {
            for output in 0..outputs {
                let value = explanation.values()[[sample, feature, output]];
                mean[[feature, output]] += value / count;
                mean_absolute[[feature, output]] += value.abs() / count;
            }
        }
    }
    let mut standard_deviation = Array2::zeros((features, outputs));
    for &sample in samples {
        for feature in 0..features {
            for output in 0..outputs {
                let delta =
                    explanation.values()[[sample, feature, output]] - mean[[feature, output]];
                standard_deviation[[feature, output]] += delta * delta / count;
            }
        }
    }
    standard_deviation.mapv_inplace(f64::sqrt);
    let quantiles = quantiles
        .iter()
        .map(|&probability| QuantileSummary {
            probability,
            values: Array2::from_shape_fn((features, outputs), |(feature, output)| {
                let mut values = samples
                    .iter()
                    .map(|&sample| explanation.values()[[sample, feature, output]])
                    .collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                interpolated_quantile(&values, probability)
            }),
        })
        .collect();
    Ok(AttributionSummary {
        sample_count: samples.len(),
        mean,
        standard_deviation,
        mean_absolute,
        quantiles,
    })
}

/// Produces one summary per cohort label in deterministic label order.
pub fn summarize_cohorts(
    explanation: &Explanation,
    labels: &[String],
    quantiles: &[f64],
) -> Result<BTreeMap<String, AttributionSummary>> {
    if labels.len() != explanation.n_samples() {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{} cohort labels", explanation.n_samples()),
            found: format!("{} labels", labels.len()),
        });
    }
    let mut cohorts = BTreeMap::<String, Vec<usize>>::new();
    for (sample, label) in labels.iter().enumerate() {
        if label.is_empty() {
            return Err(ShapError::InvalidConfiguration(
                "cohort labels cannot be empty".into(),
            ));
        }
        cohorts.entry(label.clone()).or_default().push(sample);
    }
    cohorts
        .into_iter()
        .map(|(label, samples)| Ok((label, summarize_samples(explanation, &samples, quantiles)?)))
        .collect()
}

/// Sums source-feature attributions into validated, non-overlapping groups.
pub fn group_values(explanation: &Explanation, groups: &[Vec<usize>]) -> Result<Array3<f64>> {
    explanation.validate()?;
    if groups.is_empty() || groups.iter().any(Vec::is_empty) {
        return Err(ShapError::InvalidConfiguration(
            "feature groups cannot be empty".into(),
        ));
    }
    crate::error::checked_f64_shape(
        &[
            explanation.n_samples(),
            groups.len(),
            explanation.n_outputs(),
        ],
        "grouped attributions",
    )?;
    let mut seen = vec![false; explanation.n_features()];
    for &feature in groups.iter().flatten() {
        if feature >= seen.len() {
            return Err(ShapError::InvalidFeatureIndex {
                index: feature,
                n_features: seen.len(),
            });
        }
        if seen[feature] {
            return Err(ShapError::InvalidConfiguration(
                "feature groups must not overlap".into(),
            ));
        }
        seen[feature] = true;
    }
    Ok(Array3::from_shape_fn(
        (
            explanation.n_samples(),
            groups.len(),
            explanation.n_outputs(),
        ),
        |(sample, group, output)| {
            groups[group]
                .iter()
                .map(|&feature| explanation.values()[[sample, feature, output]])
                .sum()
        },
    ))
}

/// Returns class-conditional mean absolute attribution `(features, outputs)`.
pub fn class_conditional_importance(
    explanation: &Explanation,
    classes: &[usize],
) -> Result<BTreeMap<usize, Array2<f64>>> {
    if classes.len() != explanation.n_samples() {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{} class labels", explanation.n_samples()),
            found: format!("{} labels", classes.len()),
        });
    }
    let mut members = BTreeMap::<usize, Vec<usize>>::new();
    for (sample, &class) in classes.iter().enumerate() {
        members.entry(class).or_default().push(sample);
    }
    members
        .into_iter()
        .map(|(class, samples)| {
            Ok((
                class,
                summarize_samples(explanation, &samples, &[])?.mean_absolute,
            ))
        })
        .collect()
}

fn validate_quantiles(quantiles: &[f64]) -> Result<()> {
    if quantiles
        .iter()
        .any(|probability| !probability.is_finite() || !(0.0..=1.0).contains(probability))
    {
        return Err(ShapError::InvalidConfiguration(
            "quantile probabilities must be finite and between zero and one".into(),
        ));
    }
    Ok(())
}

fn interpolated_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn explanation() -> Explanation {
        Explanation::new(
            Array3::from_shape_vec((3, 2, 1), vec![1., 2., 3., 4., 5., 6.]).unwrap(),
            Array2::zeros((3, 1)),
            Array2::zeros((3, 2)),
        )
        .unwrap()
    }

    #[test]
    fn computes_cohorts_quantiles_groups_and_class_importance() {
        let e = explanation();
        let summary = summarize(&e, &[0.5]).unwrap();
        assert_eq!(summary.mean, array![[3.], [4.]]);
        assert_eq!(summary.quantiles[0].values, array![[3.], [4.]]);
        let cohorts = summarize_cohorts(&e, &["a".into(), "b".into(), "a".into()], &[]).unwrap();
        assert_eq!(cohorts["a"].sample_count, 2);
        assert_eq!(group_values(&e, &[vec![0, 1]]).unwrap()[[2, 0, 0]], 11.0);
        assert_eq!(
            class_conditional_importance(&e, &[0, 1, 0]).unwrap()[&0],
            array![[3.], [4.]]
        );
    }
}
