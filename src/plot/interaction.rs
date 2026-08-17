use crate::{interactions::InteractionExplanation, Result, ShapError};
use ndarray::Array2;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionHeatmapData {
    pub sample: usize,
    pub output: usize,
    pub values: Array2<f64>,
    pub feature_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InteractionDependencePoint {
    pub sample: usize,
    pub feature_value: f64,
    pub interaction_value: f64,
    pub color_value: Option<f64>,
}

/// Returns one symmetric feature-by-feature interaction matrix.
pub fn heatmap_data(
    explanation: &InteractionExplanation,
    sample: usize,
    output: usize,
) -> Result<InteractionHeatmapData> {
    explanation.validate()?;
    validate(explanation, sample, 0, output)?;
    Ok(InteractionHeatmapData {
        sample,
        output,
        values: Array2::from_shape_fn(
            (explanation.n_features(), explanation.n_features()),
            |(first, second)| explanation.values()[[sample, first, second, output]],
        ),
        feature_names: explanation.feature_names().map(<[String]>::to_vec),
    })
}

/// Returns interaction strength against a feature value over all samples.
pub fn dependence_data(
    explanation: &InteractionExplanation,
    feature: usize,
    interacting_feature: usize,
    output: usize,
    color_feature: Option<usize>,
) -> Result<Vec<InteractionDependencePoint>> {
    explanation.validate()?;
    validate(explanation, 0, feature, output)?;
    if interacting_feature >= explanation.n_features() {
        return Err(ShapError::InvalidFeatureIndex {
            index: interacting_feature,
            n_features: explanation.n_features(),
        });
    }
    if let Some(index) = color_feature.filter(|&index| index >= explanation.n_features()) {
        return Err(ShapError::InvalidFeatureIndex {
            index,
            n_features: explanation.n_features(),
        });
    }
    Ok((0..explanation.n_samples())
        .map(|sample| InteractionDependencePoint {
            sample,
            feature_value: explanation.data()[[sample, feature]],
            interaction_value: explanation.values()[[sample, feature, interacting_feature, output]],
            color_value: color_feature.map(|index| explanation.data()[[sample, index]]),
        })
        .collect())
}

fn validate(
    explanation: &InteractionExplanation,
    sample: usize,
    feature: usize,
    output: usize,
) -> Result<()> {
    if sample >= explanation.n_samples() {
        return Err(ShapError::InvalidSampleIndex {
            index: sample,
            n_samples: explanation.n_samples(),
        });
    }
    if feature >= explanation.n_features() {
        return Err(ShapError::InvalidFeatureIndex {
            index: feature,
            n_features: explanation.n_features(),
        });
    }
    if output >= explanation.n_outputs() {
        return Err(ShapError::InvalidOutputIndex {
            index: output,
            n_outputs: explanation.n_outputs(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FeatureMetadata;
    use ndarray::{array, Array4};

    #[test]
    fn creates_metadata_aware_interaction_plot_data() {
        let explanation = InteractionExplanation::new(
            Array4::from_shape_vec((2, 2, 2, 1), vec![1., 0.5, 0.5, 2., 3., 1., 1., 4.]).unwrap(),
            array![[0.], [0.]],
            array![[10., 20.], [30., 40.]],
        )
        .unwrap()
        .with_feature_metadata(FeatureMetadata::new(vec!["a".into(), "b".into()]).unwrap())
        .unwrap();
        let heatmap = heatmap_data(&explanation, 1, 0).unwrap();
        assert_eq!(heatmap.values[[0, 1]], 1.0);
        assert_eq!(heatmap.feature_names.unwrap(), ["a", "b"]);
        let points = dependence_data(&explanation, 0, 1, 0, Some(1)).unwrap();
        assert_eq!(points[1].feature_value, 30.0);
        assert_eq!(points[1].color_value, Some(40.0));
    }
}
