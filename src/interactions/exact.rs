use crate::{
    coalition, evaluation::CoalitionEvaluator, EvaluationConfig, Explanation, FeatureMetadata,
    Masker, OutputMetadata, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, Array4, ArrayView2, ArrayView4, Axis};
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct InteractionExplanation {
    schema_version: u32,
    values: Array4<f64>,
    base_values: Array2<f64>,
    data: Array2<f64>,
    feature_metadata: Option<FeatureMetadata>,
    output_metadata: Option<OutputMetadata>,
}
#[derive(serde::Deserialize)]
struct InteractionExplanationPayload {
    schema_version: u32,
    values: Array4<f64>,
    base_values: Array2<f64>,
    data: Array2<f64>,
    feature_metadata: Option<FeatureMetadata>,
    output_metadata: Option<OutputMetadata>,
}
impl<'de> serde::Deserialize<'de> for InteractionExplanation {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let payload = InteractionExplanationPayload::deserialize(deserializer)?;
        if payload.schema_version != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported interaction explanation schema version {}",
                payload.schema_version
            )));
        }
        let mut value = Self::new(payload.values, payload.base_values, payload.data)
            .map_err(serde::de::Error::custom)?;
        if let Some(metadata) = payload.feature_metadata {
            value = value
                .with_feature_metadata(metadata)
                .map_err(serde::de::Error::custom)?;
        }
        if let Some(metadata) = payload.output_metadata {
            value = value
                .with_output_metadata(metadata)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(value)
    }
}
impl InteractionExplanation {
    pub fn new(values: Array4<f64>, base_values: Array2<f64>, data: Array2<f64>) -> Result<Self> {
        let (samples, features, second_features, outputs) = values.dim();
        if samples == 0 {
            return Err(ShapError::EmptyData);
        }
        if features == 0 || outputs == 0 || features != second_features {
            return Err(ShapError::DimensionMismatch {
                expected: "non-empty square interaction feature axes".into(),
                found: format!("interaction values {:?}", values.dim()),
            });
        }
        if data.dim() != (samples, features) || base_values.dim() != (samples, outputs) {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "data ({samples}, {features}) and base values ({samples}, {outputs})"
                ),
                found: format!("data {:?}, base values {:?}", data.dim(), base_values.dim()),
            });
        }
        if values
            .iter()
            .chain(base_values.iter())
            .any(|value| !value.is_finite())
        {
            return Err(ShapError::NumericalError(
                "interaction explanation contains non-finite values".into(),
            ));
        }
        for sample in 0..samples {
            for first in 0..features {
                for second in first + 1..features {
                    for output in 0..outputs {
                        let a = values[[sample, first, second, output]];
                        let b = values[[sample, second, first, output]];
                        if (a - b).abs() > 1e-10 * (1.0 + a.abs().max(b.abs())) {
                            return Err(ShapError::InvalidConfiguration(
                                "interaction matrix must be symmetric".into(),
                            ));
                        }
                    }
                }
            }
        }
        Ok(Self {
            schema_version: 1,
            values,
            base_values,
            data,
            feature_metadata: None,
            output_metadata: None,
        })
    }
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(ShapError::Unsupported(format!(
                "interaction explanation schema version {}",
                self.schema_version
            )));
        }
        let checked = Self::new(
            self.values.clone(),
            self.base_values.clone(),
            self.data.clone(),
        )?;
        if let Some(metadata) = &self.feature_metadata {
            metadata.validate()?;
            if metadata.names.len() != checked.n_features() {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{} feature metadata entries", checked.n_features()),
                    found: format!("{}", metadata.names.len()),
                });
            }
        }
        if let Some(metadata) = &self.output_metadata {
            metadata.validate()?;
            if metadata.names.len() != checked.n_outputs() {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{} output metadata entries", checked.n_outputs()),
                    found: format!("{}", metadata.names.len()),
                });
            }
        }
        Ok(())
    }
    #[cfg(feature = "json-adapters")]
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|error| ShapError::Other(format!("interaction serialization failed: {error}")))
    }
    #[cfg(feature = "json-adapters")]
    pub fn from_json(json: &str) -> Result<Self> {
        let value: Self = serde_json::from_str(json).map_err(|error| {
            ShapError::Other(format!("interaction deserialization failed: {error}"))
        })?;
        value.validate()?;
        Ok(value)
    }
    pub fn values(&self) -> ArrayView4<'_, f64> {
        self.values.view()
    }
    pub fn base_values(&self) -> ndarray::ArrayView2<'_, f64> {
        self.base_values.view()
    }
    pub fn data(&self) -> ndarray::ArrayView2<'_, f64> {
        self.data.view()
    }
    pub fn n_features(&self) -> usize {
        self.values.dim().1
    }
    pub fn n_samples(&self) -> usize {
        self.values.dim().0
    }
    pub fn n_outputs(&self) -> usize {
        self.values.dim().3
    }
    pub fn feature_metadata(&self) -> Option<&FeatureMetadata> {
        self.feature_metadata.as_ref()
    }
    pub fn output_metadata(&self) -> Option<&OutputMetadata> {
        self.output_metadata.as_ref()
    }
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_metadata.as_ref().map(|m| m.names.as_slice())
    }
    pub fn output_names(&self) -> Option<&[String]> {
        self.output_metadata.as_ref().map(|m| m.names.as_slice())
    }
    pub fn with_feature_metadata(mut self, metadata: FeatureMetadata) -> Result<Self> {
        metadata.validate()?;
        if metadata.names.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                found: format!("{} metadata entries", metadata.names.len()),
            });
        }
        self.feature_metadata = Some(metadata);
        Ok(self)
    }
    pub fn with_output_metadata(mut self, metadata: OutputMetadata) -> Result<Self> {
        metadata.validate()?;
        if metadata.names.len() != self.n_outputs() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} outputs", self.n_outputs()),
                found: format!("{} metadata entries", metadata.names.len()),
            });
        }
        self.output_metadata = Some(metadata);
        Ok(self)
    }
    pub fn select_samples(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        validate_indices(indices, self.n_samples(), "sample")?;
        let mut out = Self::new(
            self.values.select(Axis(0), indices),
            self.base_values.select(Axis(0), indices),
            self.data.select(Axis(0), indices),
        )?;
        out.feature_metadata = self.feature_metadata.clone();
        out.output_metadata = self.output_metadata.clone();
        Ok(out)
    }
    pub fn select_features(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        validate_indices(indices, self.n_features(), "feature")?;
        let values = self
            .values
            .select(Axis(1), indices)
            .select(Axis(2), indices);
        let mut out = Self::new(
            values,
            self.base_values.clone(),
            self.data.select(Axis(1), indices),
        )?;
        out.output_metadata = self.output_metadata.clone();
        out.feature_metadata = self
            .feature_metadata
            .as_ref()
            .map(|m| subset_feature_metadata(m, indices));
        Ok(out)
    }
    pub fn select_output(&self, output: usize) -> Result<Self> {
        self.validate()?;
        validate_indices(&[output], self.n_outputs(), "output")?;
        let mut out = Self::new(
            self.values.select(Axis(3), &[output]),
            self.base_values.select(Axis(1), &[output]),
            self.data.clone(),
        )?;
        out.feature_metadata = self.feature_metadata.clone();
        out.output_metadata = self.output_metadata.as_ref().map(|m| OutputMetadata {
            names: vec![m.names[output].clone()],
            kinds: m.kinds.as_ref().map(|v| vec![v[output]]),
        });
        Ok(out)
    }
    pub fn concatenate(parts: &[Self]) -> Result<Self> {
        if parts.is_empty() {
            return Err(ShapError::EmptyData);
        }
        for part in parts {
            part.validate()?;
        }
        let first = &parts[0];
        if parts.iter().any(|p| {
            p.n_features() != first.n_features()
                || p.n_outputs() != first.n_outputs()
                || p.feature_metadata != first.feature_metadata
                || p.output_metadata != first.output_metadata
        }) {
            return Err(ShapError::DimensionMismatch {
                expected: "interaction explanations with identical dimensions and metadata".into(),
                found: "incompatible interaction explanation parts".into(),
            });
        }
        let value_views = parts.iter().map(|p| p.values.view()).collect::<Vec<_>>();
        let base_views = parts
            .iter()
            .map(|p| p.base_values.view())
            .collect::<Vec<_>>();
        let data_views = parts.iter().map(|p| p.data.view()).collect::<Vec<_>>();
        let mut out = Self::new(
            ndarray::concatenate(Axis(0), &value_views)
                .map_err(|e| ShapError::Other(e.to_string()))?,
            ndarray::concatenate(Axis(0), &base_views)
                .map_err(|e| ShapError::Other(e.to_string()))?,
            ndarray::concatenate(Axis(0), &data_views)
                .map_err(|e| ShapError::Other(e.to_string()))?,
        )?;
        out.feature_metadata = first.feature_metadata.clone();
        out.output_metadata = first.output_metadata.clone();
        Ok(out)
    }
    /// Returns diagonal interaction entries `(samples, features, outputs)`.
    pub fn main_effects(&self) -> Array3<f64> {
        Array3::from_shape_fn(
            (self.n_samples(), self.n_features(), self.n_outputs()),
            |(sample, feature, output)| self.values[[sample, feature, feature, output]],
        )
    }
    /// Sums each interaction row into ordinary per-feature SHAP values.
    pub fn total_effects(&self) -> Array3<f64> {
        Array3::from_shape_fn(
            (self.n_samples(), self.n_features(), self.n_outputs()),
            |(sample, feature, output)| {
                (0..self.n_features())
                    .map(|other| self.values[[sample, feature, other, output]])
                    .sum()
            },
        )
    }
    /// Converts interaction row sums into the standard explanation type.
    pub fn to_explanation(&self) -> Result<Explanation> {
        self.validate()?;
        let mut out = Explanation::new(
            self.total_effects(),
            self.base_values.clone(),
            self.data.clone(),
        )?;
        if let Some(metadata) = &self.feature_metadata {
            out = out.with_feature_metadata(metadata.clone())?;
        }
        if let Some(metadata) = &self.output_metadata {
            out = out.with_output_metadata(metadata.clone())?;
        }
        Ok(out)
    }
    pub fn reconstructed(&self) -> Array2<f64> {
        let mut out = self.base_values.clone();
        for n in 0..self.values.dim().0 {
            for i in 0..self.values.dim().1 {
                for j in 0..self.values.dim().2 {
                    for o in 0..self.values.dim().3 {
                        out[[n, o]] += self.values[[n, i, j, o]]
                    }
                }
            }
        }
        out
    }
}

fn validate_indices(indices: &[usize], upper: usize, axis: &str) -> Result<()> {
    if indices.is_empty() {
        return Err(ShapError::EmptyData);
    }
    if let Some(&index) = indices.iter().find(|&&index| index >= upper) {
        return Err(ShapError::InvalidConfiguration(format!(
            "{axis} index {index} is out of bounds for length {upper}"
        )));
    }
    Ok(())
}

fn subset_feature_metadata(metadata: &FeatureMetadata, indices: &[usize]) -> FeatureMetadata {
    FeatureMetadata {
        names: indices.iter().map(|&i| metadata.names[i].clone()).collect(),
        display_names: metadata
            .display_names
            .as_ref()
            .map(|v| indices.iter().map(|&i| v[i].clone()).collect()),
        kinds: metadata
            .kinds
            .as_ref()
            .map(|v| indices.iter().map(|&i| v[i]).collect()),
        units: metadata
            .units
            .as_ref()
            .map(|v| indices.iter().map(|&i| v[i].clone()).collect()),
    }
}
/// Exact pairwise Shapley interactions for any prediction model and masker.
pub struct ExactInteractionExplainer<M, K> {
    model: M,
    masker: K,
    max_features: usize,
    evaluation: EvaluationConfig,
}
impl<M, K> ExactInteractionExplainer<M, K> {
    pub fn new(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            max_features: 16,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 1 << 20,
                max_model_rows: None,
            },
        }
    }
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = n;
        self
    }
    pub fn with_evaluation_config(mut self, c: EvaluationConfig) -> Self {
        self.evaluation = c;
        self
    }
}
impl<M: Predict, K: Masker> ExactInteractionExplainer<M, K> {
    pub fn explain(&self, x: ArrayView2<'_, f64>) -> Result<InteractionExplanation> {
        let m = self.masker.n_features();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features"),
                found: format!("{}", x.ncols()),
            });
        }
        if m > self.max_features || m >= 63 {
            return Err(ShapError::InvalidConfiguration(format!(
                "exact interactions support at most {} features",
                self.max_features
            )));
        }
        let masks = coalition::all(m).collect::<Vec<_>>();
        let mut probe = CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let o = probe.evaluate(x.row(0), &[0])?[0].len();
        crate::error::checked_f64_shape(&[x.nrows(), m, m, o], "interaction explanation")?;
        let mut values = Array4::zeros((x.nrows(), m, m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        let factorial = (0..=m)
            .scan(1., |v, k| {
                if k > 0 {
                    *v *= k as f64
                }
                Some(*v)
            })
            .collect::<Vec<_>>();
        for n in 0..x.nrows() {
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let cache = evaluator.evaluate(x.row(n), &masks)?;
            for o in 0..o {
                bases[[n, o]] = cache[0][o]
            }
            let mut shap = Array3::<f64>::zeros((m, 1, o));
            for i in 0..m {
                for mask in masks.iter().copied().filter(|z| z & (1 << i) == 0) {
                    let s = mask.count_ones() as usize;
                    let w = factorial[s] * factorial[m - s - 1] / factorial[m];
                    for k in 0..o {
                        shap[[i, 0, k]] +=
                            w * (cache[(mask | (1 << i)) as usize][k] - cache[mask as usize][k])
                    }
                }
            }
            for i in 0..m {
                for j in i + 1..m {
                    for mask in masks
                        .iter()
                        .copied()
                        .filter(|z| z & (1 << i) == 0 && z & (1 << j) == 0)
                    {
                        let s = mask.count_ones() as usize;
                        let w = factorial[s] * factorial[m - s - 2] / (2. * factorial[m - 1]);
                        for k in 0..o {
                            let d = cache[(mask | (1 << i) | (1 << j)) as usize][k]
                                - cache[(mask | (1 << i)) as usize][k]
                                - cache[(mask | (1 << j)) as usize][k]
                                + cache[mask as usize][k];
                            values[[n, i, j, k]] += w * d;
                            values[[n, j, i, k]] += w * d
                        }
                    }
                }
            }
            for i in 0..m {
                for k in 0..o {
                    values[[n, i, i, k]] = shap[[i, 0, k]]
                        - (0..m)
                            .filter(|&j| j != i)
                            .map(|j| values[[n, i, j, k]])
                            .sum::<f64>()
                }
            }
        }
        InteractionExplanation::new(values, bases, x.to_owned())
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedMasker, FnModel};
    use ndarray::{array, Axis};
    #[test]
    fn exact_interactions_are_symmetric_and_additive() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| r[0] * r[1] + r[0])
                .insert_axis(Axis(1)))
        });
        let e = ExactInteractionExplainer::new(model, FixedMasker::new(array![0., 0.]).unwrap())
            .explain(array![[2., 3.]].view())
            .unwrap();
        assert!((e.values()[[0, 0, 1, 0]] - e.values()[[0, 1, 0, 0]]).abs() < 1e-12);
        assert!((e.reconstructed()[[0, 0]] - 8.).abs() < 1e-12);
        let ordinary = e.to_explanation().unwrap();
        assert!((ordinary.reconstructed()[[0, 0]] - 8.).abs() < 1e-12);
        assert_eq!(e.main_effects().dim(), (1, 2, 1));
        for feature in 0..2 {
            let row_sum = (0..2)
                .map(|other| e.values()[[0, feature, other, 0]])
                .sum::<f64>();
            assert!((ordinary.values()[[0, feature, 0]] - row_sum).abs() < 1e-12);
        }
    }
    #[test]
    fn validates_shape_symmetry_and_json_round_trip() {
        let asymmetric = Array4::from_shape_vec((1, 2, 2, 1), vec![0., 1., 2., 0.]).unwrap();
        assert!(InteractionExplanation::new(asymmetric, array![[0.]], array![[0., 0.]]).is_err());

        let valid = InteractionExplanation::new(
            Array4::from_shape_vec((1, 2, 2, 1), vec![1., 0.5, 0.5, 2.]).unwrap(),
            array![[3.]],
            array![[4., 5.]],
        )
        .unwrap();
        assert_eq!(valid.schema_version(), 1);
        #[cfg(feature = "json-adapters")]
        {
            let decoded = InteractionExplanation::from_json(&valid.to_json().unwrap()).unwrap();
            assert_eq!(decoded, valid);
            assert_eq!(decoded.schema_version(), 1);
        }
    }

    #[test]
    fn metadata_selection_concatenation_and_conversion_are_consistent() {
        use crate::{FeatureMetadata, OutputMetadata};
        let values = Array4::from_shape_fn((2, 2, 2, 2), |(n, i, j, o)| {
            if i == j {
                (n * 8 + i * 2 + o + 1) as f64
            } else {
                0.0
            }
        });
        let explanation = InteractionExplanation::new(
            values,
            array![[0., 1.], [2., 3.]],
            array![[10., 20.], [30., 40.]],
        )
        .unwrap()
        .with_feature_metadata(FeatureMetadata::new(vec!["a".into(), "b".into()]).unwrap())
        .unwrap()
        .with_output_metadata(OutputMetadata::new(vec!["x".into(), "y".into()]).unwrap())
        .unwrap();
        let selected = explanation
            .select_samples(&[1])
            .unwrap()
            .select_features(&[1])
            .unwrap()
            .select_output(0)
            .unwrap();
        assert_eq!(selected.values().dim(), (1, 1, 1, 1));
        assert_eq!(selected.feature_names().unwrap(), &["b"]);
        assert_eq!(
            selected.to_explanation().unwrap().output_names().unwrap(),
            &["x"]
        );
        let halves = [
            explanation.select_samples(&[0]).unwrap(),
            explanation.select_samples(&[1]).unwrap(),
        ];
        assert_eq!(
            InteractionExplanation::concatenate(&halves).unwrap(),
            explanation
        );
    }
}
