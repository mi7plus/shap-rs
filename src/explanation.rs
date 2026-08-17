use crate::{FeatureMetadata, OutputMetadata, Result, ShapError};
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use serde::{Deserialize, Deserializer, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AttributionSemantics {
    #[default]
    Unspecified,
    Interventional,
    Conditional,
    TreePathDependent,
    CausalAsymmetric,
}
/// SHAP values `(samples, features, outputs)`, reference values, and explained data.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Explanation {
    schema_version: u32,
    values: Array3<f64>,
    base_values: Array2<f64>,
    data: Array2<f64>,
    feature_metadata: Option<FeatureMetadata>,
    output_metadata: Option<OutputMetadata>,
    semantics: AttributionSemantics,
}
/// An explanation accompanied by per-attribution Monte-Carlo standard errors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UncertainExplanation {
    explanation: Explanation,
    standard_errors: Array3<f64>,
    repeats: usize,
}
#[derive(Deserialize)]
struct ExplanationPayload {
    schema_version: u32,
    values: Array3<f64>,
    base_values: Array2<f64>,
    data: Array2<f64>,
    feature_metadata: Option<FeatureMetadata>,
    output_metadata: Option<OutputMetadata>,
    #[serde(default)]
    semantics: AttributionSemantics,
}
impl<'de> Deserialize<'de> for Explanation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let payload = ExplanationPayload::deserialize(deserializer)?;
        if payload.schema_version != 1 {
            return Err(serde::de::Error::custom(format!(
                "unsupported explanation schema version {}",
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
        value.semantics = payload.semantics;
        Ok(value)
    }
}
#[derive(Deserialize)]
struct UncertainExplanationPayload {
    explanation: Explanation,
    standard_errors: Array3<f64>,
    repeats: usize,
}
impl<'de> Deserialize<'de> for UncertainExplanation {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let payload = UncertainExplanationPayload::deserialize(deserializer)?;
        Self::new(
            payload.explanation,
            payload.standard_errors,
            payload.repeats,
        )
        .map_err(serde::de::Error::custom)
    }
}
impl UncertainExplanation {
    pub fn concatenate(parts: &[Self]) -> Result<Self> {
        if parts.is_empty() {
            return Err(ShapError::EmptyData);
        }
        for part in parts {
            part.validate()?;
        }
        let repeats = parts[0].repeats;
        if parts.iter().any(|part| part.repeats != repeats) {
            return Err(ShapError::InvalidConfiguration(
                "uncertain explanations must have identical repeat counts".into(),
            ));
        }
        let explanations = parts
            .iter()
            .map(|p| p.explanation.clone())
            .collect::<Vec<_>>();
        let error_views = parts
            .iter()
            .map(|p| p.standard_errors.view())
            .collect::<Vec<_>>();
        Self::new(
            Explanation::concatenate(&explanations)?,
            ndarray::concatenate(Axis(0), &error_views)
                .map_err(|error| ShapError::Other(error.to_string()))?,
            repeats,
        )
    }
    pub fn new(
        explanation: Explanation,
        standard_errors: Array3<f64>,
        repeats: usize,
    ) -> Result<Self> {
        explanation.validate()?;
        if standard_errors.dim() != explanation.values.dim() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{:?} standard errors", explanation.values.dim()),
                found: format!("{:?}", standard_errors.dim()),
            });
        }
        if repeats < 2 || standard_errors.iter().any(|v| !v.is_finite() || *v < 0.) {
            return Err(ShapError::InvalidConfiguration(
                "uncertainty requires at least two valid repeated estimates".into(),
            ));
        }
        Ok(Self {
            explanation,
            standard_errors,
            repeats,
        })
    }
    pub fn validate(&self) -> Result<()> {
        self.explanation.validate()?;
        if self.standard_errors.dim() != self.explanation.values.dim() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{:?} standard errors", self.explanation.values.dim()),
                found: format!("{:?}", self.standard_errors.dim()),
            });
        }
        if self.repeats < 2
            || self
                .standard_errors
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ShapError::InvalidConfiguration(
                "uncertainty requires at least two valid repeated estimates".into(),
            ));
        }
        Ok(())
    }
    pub fn explanation(&self) -> &Explanation {
        &self.explanation
    }
    pub fn standard_errors(&self) -> ArrayView3<'_, f64> {
        self.standard_errors.view()
    }
    pub fn repeats(&self) -> usize {
        self.repeats
    }
    pub fn into_explanation(self) -> Explanation {
        self.explanation
    }
    pub fn select_samples(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        validate_indices(indices, self.explanation.n_samples(), "sample")?;
        Self::new(
            self.explanation.select_samples(indices)?,
            self.standard_errors.select(Axis(0), indices),
            self.repeats,
        )
    }
    pub fn select_features(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        validate_indices(indices, self.explanation.n_features(), "feature")?;
        Self::new(
            self.explanation.select_features(indices)?,
            self.standard_errors.select(Axis(1), indices),
            self.repeats,
        )
    }
    pub fn select_output(&self, output: usize) -> Result<Self> {
        self.validate()?;
        validate_indices(&[output], self.explanation.n_outputs(), "output")?;
        Self::new(
            self.explanation.select_output(output)?,
            self.standard_errors.select(Axis(2), &[output]),
            self.repeats,
        )
    }
    pub fn confidence_interval(&self, z_score: f64) -> Result<(Array3<f64>, Array3<f64>)> {
        self.validate()?;
        if !z_score.is_finite() || z_score < 0. {
            return Err(ShapError::InvalidConfiguration(
                "confidence interval z-score must be finite and non-negative".into(),
            ));
        }
        let delta = self.standard_errors.mapv(|e| e * z_score);
        Ok((
            &self.explanation.values - &delta,
            &self.explanation.values + &delta,
        ))
    }
}
impl Explanation {
    pub fn concatenate(parts: &[Explanation]) -> Result<Self> {
        if parts.is_empty() {
            return Err(ShapError::EmptyData);
        }
        for part in parts {
            part.validate()?;
        }
        let features = parts[0].n_features();
        let outputs = parts[0].n_outputs();
        if parts.iter().any(|e| {
            e.n_features() != features
                || e.n_outputs() != outputs
                || e.feature_metadata != parts[0].feature_metadata
                || e.output_metadata != parts[0].output_metadata
                || e.semantics != parts[0].semantics
        }) {
            return Err(ShapError::DimensionMismatch {
                expected: "explanations with identical feature/output dimensions and metadata"
                    .into(),
                found: "incompatible explanation parts".into(),
            });
        }
        let value_views = parts.iter().map(|e| e.values.view()).collect::<Vec<_>>();
        let base_views = parts
            .iter()
            .map(|e| e.base_values.view())
            .collect::<Vec<_>>();
        let data_views = parts.iter().map(|e| e.data.view()).collect::<Vec<_>>();
        let values = ndarray::concatenate(Axis(0), &value_views)
            .map_err(|e| ShapError::Other(e.to_string()))?;
        let bases = ndarray::concatenate(Axis(0), &base_views)
            .map_err(|e| ShapError::Other(e.to_string()))?;
        let data = ndarray::concatenate(Axis(0), &data_views)
            .map_err(|e| ShapError::Other(e.to_string()))?;
        let mut result = Self::new(values, bases, data)?;
        result.feature_metadata = parts[0].feature_metadata.clone();
        result.output_metadata = parts[0].output_metadata.clone();
        result.semantics = parts[0].semantics;
        Ok(result)
    }
    pub fn new(values: Array3<f64>, base_values: Array2<f64>, data: Array2<f64>) -> Result<Self> {
        let (vn, vf, vo) = values.dim();
        if vn == 0 {
            return Err(ShapError::EmptyData);
        }
        if vf == 0 || vo == 0 {
            return Err(ShapError::InvalidConfiguration(
                "an explanation must contain at least one feature and one output".into(),
            ));
        }
        if (vn, vf) != (data.nrows(), data.ncols())
            || (vn, vo) != (base_values.nrows(), base_values.ncols())
        {
            return Err(ShapError::DimensionMismatch {
                expected: "consistent sample/feature/output dimensions".into(),
                found: format!(
                    "values {:?}, base {:?}, data {:?}",
                    values.dim(),
                    base_values.dim(),
                    data.dim()
                ),
            });
        }
        if values
            .iter()
            .chain(base_values.iter())
            .any(|v| !v.is_finite())
        {
            return Err(ShapError::NumericalError(
                "explanation contains non-finite SHAP or base values".into(),
            ));
        }
        Ok(Self {
            schema_version: 1,
            values,
            base_values,
            data,
            feature_metadata: None,
            output_metadata: None,
            semantics: AttributionSemantics::Unspecified,
        })
    }
    pub fn values(&self) -> ArrayView3<'_, f64> {
        self.values.view()
    }
    pub fn schema_version(&self) -> u32 {
        self.schema_version
    }
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(ShapError::Unsupported(format!(
                "explanation schema version {}",
                self.schema_version
            )));
        }
        let checked = Self::new(
            self.values.clone(),
            self.base_values.clone(),
            self.data.clone(),
        )?;
        if let Some(m) = &self.feature_metadata {
            m.validate()?;
            if m.names.len() != checked.n_features() {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{} feature metadata entries", checked.n_features()),
                    found: format!("{}", m.names.len()),
                });
            }
        }
        if let Some(m) = &self.output_metadata {
            m.validate()?;
            if m.names.len() != checked.n_outputs() {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{} output metadata entries", checked.n_outputs()),
                    found: format!("{}", m.names.len()),
                });
            }
        }
        Ok(())
    }
    #[cfg(feature = "json-adapters")]
    pub fn to_json(&self) -> Result<String> {
        self.validate()?;
        serde_json::to_string(self)
            .map_err(|e| ShapError::Other(format!("explanation serialization failed: {e}")))
    }
    #[cfg(feature = "json-adapters")]
    pub fn from_json(json: &str) -> Result<Self> {
        let value: Self = serde_json::from_str(json)
            .map_err(|e| ShapError::Other(format!("explanation deserialization failed: {e}")))?;
        value.validate()?;
        Ok(value)
    }
    pub fn base_values(&self) -> ArrayView2<'_, f64> {
        self.base_values.view()
    }
    pub fn data(&self) -> ArrayView2<'_, f64> {
        self.data.view()
    }
    pub fn n_samples(&self) -> usize {
        self.values.dim().0
    }
    pub fn n_features(&self) -> usize {
        self.values.dim().1
    }
    pub fn n_outputs(&self) -> usize {
        self.values.dim().2
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
        out.semantics = self.semantics;
        Ok(out)
    }
    pub fn select_features(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        validate_indices(indices, self.n_features(), "feature")?;
        let mut out = Self::new(
            self.values.select(Axis(1), indices),
            self.base_values.clone(),
            self.data.select(Axis(1), indices),
        )?;
        out.output_metadata = self.output_metadata.clone();
        out.semantics = self.semantics;
        if let Some(m) = &self.feature_metadata {
            out.feature_metadata = Some(FeatureMetadata {
                names: indices.iter().map(|&j| m.names[j].clone()).collect(),
                display_names: m
                    .display_names
                    .as_ref()
                    .map(|v| indices.iter().map(|&j| v[j].clone()).collect()),
                kinds: m
                    .kinds
                    .as_ref()
                    .map(|v| indices.iter().map(|&j| v[j]).collect()),
                units: m
                    .units
                    .as_ref()
                    .map(|v| indices.iter().map(|&j| v[j].clone()).collect()),
            })
        }
        Ok(out)
    }
    pub fn select_output(&self, output: usize) -> Result<Self> {
        self.validate()?;
        validate_indices(&[output], self.n_outputs(), "output")?;
        let mut out = Self::new(
            self.values.select(Axis(2), &[output]),
            self.base_values.select(Axis(1), &[output]),
            self.data.clone(),
        )?;
        out.feature_metadata = self.feature_metadata.clone();
        out.semantics = self.semantics;
        if let Some(m) = &self.output_metadata {
            out.output_metadata = Some(OutputMetadata {
                names: vec![m.names[output].clone()],
                kinds: m.kinds.as_ref().map(|v| vec![v[output]]),
            })
        }
        Ok(out)
    }
    pub fn with_feature_names(self, n: Vec<String>) -> Result<Self> {
        if n.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} names", self.n_features()),
                found: format!("{} names", n.len()),
            });
        }
        self.with_feature_metadata(FeatureMetadata::new(n)?)
    }
    pub fn with_output_names(self, n: Vec<String>) -> Result<Self> {
        if n.len() != self.n_outputs() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} names", self.n_outputs()),
                found: format!("{} names", n.len()),
            });
        }
        self.with_output_metadata(OutputMetadata::new(n)?)
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
    pub fn with_semantics(mut self, semantics: AttributionSemantics) -> Self {
        self.semantics = semantics;
        self
    }
    pub fn semantics(&self) -> AttributionSemantics {
        self.semantics
    }
    pub fn feature_names(&self) -> Option<&[String]> {
        self.feature_metadata.as_ref().map(|m| m.names.as_slice())
    }
    pub fn output_names(&self) -> Option<&[String]> {
        self.output_metadata.as_ref().map(|m| m.names.as_slice())
    }
    pub fn feature_metadata(&self) -> Option<&FeatureMetadata> {
        self.feature_metadata.as_ref()
    }
    pub fn output_metadata(&self) -> Option<&OutputMetadata> {
        self.output_metadata.as_ref()
    }
    pub fn reconstructed(&self) -> Array2<f64> {
        let mut r = self.base_values.clone();
        for i in 0..self.n_samples() {
            for j in 0..self.n_features() {
                for o in 0..self.n_outputs() {
                    r[[i, o]] += self.values[[i, j, o]]
                }
            }
        }
        r
    }
}
fn validate_indices(indices: &[usize], len: usize, label: &str) -> Result<()> {
    if indices.is_empty() {
        return Err(ShapError::InvalidConfiguration(format!(
            "{label} selection cannot be empty"
        )));
    }
    let mut seen = std::collections::HashSet::new();
    for &i in indices {
        if i >= len {
            return Err(match label {
                "sample" => ShapError::InvalidSampleIndex {
                    index: i,
                    n_samples: len,
                },
                "feature" => ShapError::InvalidFeatureIndex {
                    index: i,
                    n_features: len,
                },
                "output" => ShapError::InvalidOutputIndex {
                    index: i,
                    n_outputs: len,
                },
                _ => ShapError::InvalidConfiguration(format!(
                    "{label} index {i} is out of bounds for {len}"
                )),
            });
        }
        if !seen.insert(i) {
            return Err(ShapError::InvalidConfiguration(format!(
                "duplicate {label} index {i}"
            )));
        }
    }
    Ok(())
}
#[cfg(all(test, feature = "json-adapters"))]
mod tests {
    use super::*;
    #[test]
    fn json_round_trip_preserves_schema() {
        let e = Explanation::new(
            Array3::zeros((1, 2, 1)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 2)),
        )
        .unwrap()
        .with_feature_names(vec!["a".into(), "b".into()])
        .unwrap();
        let decoded = Explanation::from_json(&e.to_json().unwrap()).unwrap();
        assert_eq!(decoded, e);
    }

    #[test]
    fn rejects_empty_explanation_axes() {
        assert!(matches!(
            Explanation::new(
                Array3::zeros((0, 1, 1)),
                Array2::zeros((0, 1)),
                Array2::zeros((0, 1))
            ),
            Err(ShapError::EmptyData)
        ));
        assert!(Explanation::new(
            Array3::zeros((1, 0, 1)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 0)),
        )
        .is_err());
        assert!(Explanation::new(
            Array3::zeros((1, 1, 0)),
            Array2::zeros((1, 0)),
            Array2::zeros((1, 1)),
        )
        .is_err());
    }
}
#[cfg(test)]
mod concatenate_tests {
    use super::*;
    #[test]
    fn concatenates_sample_batches() {
        let a = Explanation::new(
            Array3::zeros((1, 2, 1)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 2)),
        )
        .unwrap();
        let b = Explanation::new(
            Array3::ones((2, 2, 1)),
            Array2::ones((2, 1)),
            Array2::ones((2, 2)),
        )
        .unwrap();
        let e = Explanation::concatenate(&[a, b]).unwrap();
        assert_eq!(e.values().dim(), (3, 2, 1));
        assert_eq!(e.values()[[2, 1, 0]], 1.);
    }
}
#[cfg(test)]
mod uncertainty_tests {
    use super::*;

    #[test]
    fn confidence_intervals_revalidate_deserialized_style_state() {
        let explanation = Explanation::new(
            Array3::zeros((1, 1, 1)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 1)),
        )
        .unwrap();
        let malformed = UncertainExplanation {
            explanation,
            standard_errors: Array3::from_elem((1, 1, 1), f64::NAN),
            repeats: 1,
        };
        assert!(malformed.confidence_interval(1.96).is_err());
    }

    #[test]
    fn selections_and_concatenation_preserve_uncertainty() {
        let explanation = Explanation::new(
            Array3::from_shape_vec((2, 2, 2), (0..8).map(f64::from).collect()).unwrap(),
            ndarray::array![[0., 1.], [2., 3.]],
            ndarray::array![[10., 20.], [30., 40.]],
        )
        .unwrap()
        .with_feature_names(vec!["a".into(), "b".into()])
        .unwrap()
        .with_output_names(vec!["x".into(), "y".into()])
        .unwrap();
        let uncertain =
            UncertainExplanation::new(explanation, Array3::from_elem((2, 2, 2), 0.5), 4).unwrap();
        let selected = uncertain
            .select_samples(&[1])
            .unwrap()
            .select_features(&[0])
            .unwrap()
            .select_output(1)
            .unwrap();
        assert_eq!(selected.standard_errors().dim(), (1, 1, 1));
        assert_eq!(selected.explanation().feature_names().unwrap(), &["a"]);
        let halves = [
            uncertain.select_samples(&[0]).unwrap(),
            uncertain.select_samples(&[1]).unwrap(),
        ];
        assert_eq!(
            UncertainExplanation::concatenate(&halves).unwrap(),
            uncertain
        );
    }
}
#[cfg(test)]
mod selection_tests {
    use super::*;
    #[test]
    fn selections_preserve_matching_metadata() {
        let e = Explanation::new(
            Array3::zeros((2, 3, 2)),
            Array2::zeros((2, 2)),
            Array2::zeros((2, 3)),
        )
        .unwrap()
        .with_feature_names(vec!["a".into(), "b".into(), "c".into()])
        .unwrap()
        .with_output_names(vec!["left".into(), "right".into()])
        .unwrap();
        let selected = e
            .select_samples(&[1])
            .unwrap()
            .select_features(&[2, 0])
            .unwrap()
            .select_output(1)
            .unwrap();
        assert_eq!(selected.values().dim(), (1, 2, 1));
        assert_eq!(selected.feature_names().unwrap(), ["c", "a"]);
        assert_eq!(selected.output_names().unwrap(), ["right"]);
    }

    #[test]
    fn selections_report_typed_bounds_errors() {
        let e = Explanation::new(
            Array3::zeros((1, 2, 1)),
            Array2::zeros((1, 1)),
            Array2::zeros((1, 2)),
        )
        .unwrap();
        assert!(matches!(
            e.select_samples(&[1]),
            Err(ShapError::InvalidSampleIndex { .. })
        ));
        assert!(matches!(
            e.select_features(&[2]),
            Err(ShapError::InvalidFeatureIndex { .. })
        ));
        assert!(matches!(
            e.select_output(1),
            Err(ShapError::InvalidOutputIndex { .. })
        ));
    }
}
