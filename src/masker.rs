use crate::{Background, Result, ShapError};
use ndarray::{Array1, Array2, ArrayView1, Axis};
pub trait Masker {
    /// Number of coalition features exposed to an explainer.
    fn n_features(&self) -> usize;
    /// Number of columns expected in model input samples.
    ///
    /// This differs from [`Masker::n_features`] for grouped maskers.
    fn n_input_features(&self) -> usize {
        self.n_features()
    }
    /// Values stored on the feature axis of the resulting explanation.
    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if samples.ncols() != self.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} input features", self.n_input_features()),
                found: format!("{}", samples.ncols()),
            });
        }
        Ok(samples.to_owned())
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>>;
}
impl<T: Masker + ?Sized> Masker for &T {
    fn n_features(&self) -> usize {
        (**self).n_features()
    }
    fn n_input_features(&self) -> usize {
        (**self).n_input_features()
    }
    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        (**self).attribution_data(samples)
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        (**self).mask(sample, present)
    }
}

/// Makes groups of source columns behave as single coalition features.
///
/// `groups` must form an exact partition of the wrapped masker's input
/// columns. Returned SHAP values therefore have one feature axis entry per
/// group, while the model continues to receive its original columns.
#[derive(Debug, Clone)]
pub struct GroupedMasker<K> {
    inner: K,
    groups: Vec<Vec<usize>>,
}

impl<K: Masker> GroupedMasker<K> {
    pub fn new(inner: K, groups: Vec<Vec<usize>>) -> Result<Self> {
        let n = inner.n_features();
        if n != inner.n_input_features() {
            return Err(ShapError::InvalidConfiguration(
                "nested grouped maskers are not supported".into(),
            ));
        }
        if groups.is_empty() || groups.iter().any(Vec::is_empty) {
            return Err(ShapError::InvalidConfiguration(
                "feature groups must be non-empty".into(),
            ));
        }
        let mut seen = vec![false; n];
        for &column in groups.iter().flatten() {
            if column >= n {
                return Err(ShapError::InvalidFeatureIndex {
                    index: column,
                    n_features: n,
                });
            }
            if std::mem::replace(&mut seen[column], true) {
                return Err(ShapError::InvalidConfiguration(format!(
                    "source column {column} belongs to more than one feature group"
                )));
            }
        }
        if seen.iter().any(|included| !included) {
            return Err(ShapError::InvalidConfiguration(
                "feature groups must cover every source column exactly once".into(),
            ));
        }
        Ok(Self { inner, groups })
    }

    pub fn inner(&self) -> &K {
        &self.inner
    }

    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
}

impl<K: Masker> Masker for GroupedMasker<K> {
    fn n_features(&self) -> usize {
        self.groups.len()
    }

    fn n_input_features(&self) -> usize {
        self.inner.n_input_features()
    }

    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if samples.ncols() != self.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} input features", self.n_input_features()),
                found: format!("{}", samples.ncols()),
            });
        }
        let mut grouped = Array2::zeros((samples.nrows(), self.groups.len()));
        for (group_index, group) in self.groups.iter().enumerate() {
            for row in 0..samples.nrows() {
                grouped[[row, group_index]] = group
                    .iter()
                    .map(|&column| samples[[row, column]])
                    .sum::<f64>()
                    / group.len() as f64;
            }
        }
        Ok(grouped)
    }

    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        if sample.len() != self.n_input_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "{} input columns and {} feature groups",
                    self.n_input_features(),
                    self.n_features()
                ),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut expanded = vec![false; self.inner.n_features()];
        for (group, &on) in self.groups.iter().zip(present) {
            for &column in group {
                expanded[column] = on;
            }
        }
        self.inner.mask(sample, &expanded)
    }
}
/// Adapts a closure into a masker, enabling conditional, sparse, structured,
/// text, or image masking without implementing a named type.
pub struct FnMasker<F> {
    n_features: usize,
    mask_fn: F,
}
impl<F> FnMasker<F> {
    pub fn new(n_features: usize, mask_fn: F) -> Result<Self> {
        if n_features == 0 {
            return Err(ShapError::InvalidConfiguration(
                "masker must expose at least one feature".into(),
            ));
        }
        Ok(Self {
            n_features,
            mask_fn,
        })
    }
}
impl<F> Masker for FnMasker<F>
where
    F: Fn(ArrayView1<'_, f64>, &[bool]) -> Result<Array2<f64>>,
{
    fn n_features(&self) -> usize {
        self.n_features
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        if sample.len() != self.n_features || present.len() != self.n_features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        (self.mask_fn)(sample, present)
    }
}
/// Replaces absent features with a single fixed reference vector.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "FixedMaskerPayload")]
pub struct FixedMasker {
    reference: Array1<f64>,
}
#[derive(serde::Deserialize)]
struct FixedMaskerPayload {
    reference: Array1<f64>,
}
impl TryFrom<FixedMaskerPayload> for FixedMasker {
    type Error = ShapError;
    fn try_from(payload: FixedMaskerPayload) -> Result<Self> {
        Self::new(payload.reference)
    }
}
impl FixedMasker {
    pub fn new(reference: Array1<f64>) -> Result<Self> {
        if reference.is_empty() {
            return Err(ShapError::InvalidConfiguration(
                "fixed masker reference cannot be empty".into(),
            ));
        }
        Ok(Self { reference })
    }
    pub fn reference(&self) -> ndarray::ArrayView1<'_, f64> {
        self.reference.view()
    }
    pub fn validate(&self) -> Result<()> {
        if self.reference.is_empty() {
            return Err(ShapError::InvalidConfiguration(
                "fixed masker reference cannot be empty".into(),
            ));
        }
        Ok(())
    }
}
impl Masker for FixedMasker {
    fn n_features(&self) -> usize {
        self.reference.len()
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        self.validate()?;
        if sample.len() != self.n_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut row = self.reference.clone();
        for (j, &on) in present.iter().enumerate() {
            if on {
                row[j] = sample[j]
            }
        }
        Array2::from_shape_vec((1, row.len()), row.to_vec())
            .map_err(|e| ShapError::MaskerError(e.to_string()))
    }
}

/// Masks numeric token IDs with a configured mask token. Positions marked
/// immutable (for example BOS/EOS) are always retained.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "TextMaskerPayload")]
pub struct TextMasker {
    n_tokens: usize,
    mask_token: f64,
    immutable: Vec<bool>,
}
#[derive(serde::Deserialize)]
struct TextMaskerPayload {
    n_tokens: usize,
    mask_token: f64,
    immutable: Vec<bool>,
}
impl TryFrom<TextMaskerPayload> for TextMasker {
    type Error = ShapError;
    fn try_from(payload: TextMaskerPayload) -> Result<Self> {
        let masker = Self {
            n_tokens: payload.n_tokens,
            mask_token: payload.mask_token,
            immutable: payload.immutable,
        };
        masker.validate()?;
        Ok(masker)
    }
}
impl TextMasker {
    pub fn new(n_tokens: usize, mask_token: f64) -> Result<Self> {
        if n_tokens == 0 || !mask_token.is_finite() {
            return Err(ShapError::InvalidConfiguration(
                "text masker requires tokens and a finite mask token".into(),
            ));
        }
        Ok(Self {
            n_tokens,
            mask_token,
            immutable: vec![false; n_tokens],
        })
    }
    pub fn with_immutable_positions(mut self, positions: &[usize]) -> Result<Self> {
        for &j in positions {
            if j >= self.n_tokens {
                return Err(ShapError::InvalidFeatureIndex {
                    index: j,
                    n_features: self.n_tokens,
                });
            }
            self.immutable[j] = true
        }
        Ok(self)
    }
    pub fn validate(&self) -> Result<()> {
        if self.n_tokens == 0
            || !self.mask_token.is_finite()
            || self.immutable.len() != self.n_tokens
        {
            return Err(ShapError::InvalidConfiguration(
                "text masker has invalid token metadata".into(),
            ));
        }
        Ok(())
    }
}
impl Masker for TextMasker {
    fn n_features(&self) -> usize {
        self.n_tokens
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        self.validate()?;
        if sample.len() != self.n_tokens || present.len() != self.n_tokens {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} tokens", self.n_tokens),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let row = (0..self.n_tokens)
            .map(|j| {
                if present[j] || self.immutable[j] {
                    sample[j]
                } else {
                    self.mask_token
                }
            })
            .collect::<Vec<_>>();
        Array2::from_shape_vec((1, self.n_tokens), row)
            .map_err(|e| ShapError::MaskerError(e.to_string()))
    }
}

/// Masks a flattened image against a reference image. Each channel value is
/// an independently explainable feature; dimensions are retained as metadata.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "ImageMaskerPayload")]
pub struct ImageMasker {
    width: usize,
    height: usize,
    channels: usize,
    reference: Array1<f64>,
}
#[derive(serde::Deserialize)]
struct ImageMaskerPayload {
    width: usize,
    height: usize,
    channels: usize,
    reference: Array1<f64>,
}
impl TryFrom<ImageMaskerPayload> for ImageMasker {
    type Error = ShapError;
    fn try_from(payload: ImageMaskerPayload) -> Result<Self> {
        Self::new(
            payload.width,
            payload.height,
            payload.channels,
            payload.reference,
        )
    }
}
impl ImageMasker {
    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        reference: Array1<f64>,
    ) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|x| x.checked_mul(channels))
            .ok_or_else(|| ShapError::InvalidConfiguration("image dimensions overflow".into()))?;
        if expected == 0 || reference.len() != expected {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{expected} image values"),
                found: format!("{}", reference.len()),
            });
        }
        Ok(Self {
            width,
            height,
            channels,
            reference,
        })
    }
    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.channels)
    }
    pub fn validate(&self) -> Result<()> {
        let expected = self
            .width
            .checked_mul(self.height)
            .and_then(|value| value.checked_mul(self.channels))
            .ok_or_else(|| ShapError::InvalidConfiguration("image dimensions overflow".into()))?;
        if expected == 0 || self.reference.len() != expected {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{expected} image values"),
                found: format!("{}", self.reference.len()),
            });
        }
        Ok(())
    }
}
impl Masker for ImageMasker {
    fn n_features(&self) -> usize {
        self.reference.len()
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        self.validate()?;
        if sample.len() != self.n_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} image values", self.n_features()),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut row = self.reference.clone();
        for (j, &on) in present.iter().enumerate() {
            if on {
                row[j] = sample[j]
            }
        }
        Array2::from_shape_vec((1, row.len()), row.to_vec())
            .map_err(|e| ShapError::MaskerError(e.to_string()))
    }
}
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IndependentMasker {
    background: Background,
}
impl IndependentMasker {
    pub fn new(background: Background) -> Self {
        Self { background }
    }
    pub fn background(&self) -> &Background {
        &self.background
    }
    pub fn baseline(&self) -> Result<Array1<f64>> {
        self.background.validate()?;
        self.background
            .data()
            .mean_axis(Axis(0))
            .ok_or(ShapError::EmptyBackground)
    }
}
impl Masker for IndependentMasker {
    fn n_features(&self) -> usize {
        self.background.n_features()
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        self.background.validate()?;
        if sample.len() != self.n_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut out = self.background.data().to_owned();
        for (j, &on) in present.iter().enumerate() {
            if on {
                out.column_mut(j).fill(sample[j])
            }
        }
        Ok(out)
    }
}

/// Empirical conditional tabular masker using nearest background rows on the
/// currently present features. Categorical columns use exact-match distance;
/// numerical columns use variance-scaled squared distance.
#[derive(Debug, Clone)]
pub struct ConditionalTabularMasker {
    background: Background,
    categorical: Vec<bool>,
    scales: Vec<f64>,
    neighbors: usize,
}

impl ConditionalTabularMasker {
    pub fn new(
        background: Background,
        categorical_features: &[usize],
        neighbors: usize,
    ) -> Result<Self> {
        background.validate()?;
        if neighbors == 0 {
            return Err(ShapError::InvalidConfiguration(
                "conditional tabular neighbors must be positive".into(),
            ));
        }
        let mut categorical = vec![false; background.n_features()];
        for &feature in categorical_features {
            if feature >= categorical.len() {
                return Err(ShapError::InvalidFeatureIndex {
                    index: feature,
                    n_features: categorical.len(),
                });
            }
            if categorical[feature] {
                return Err(ShapError::InvalidConfiguration(
                    "categorical feature indices must be unique".into(),
                ));
            }
            categorical[feature] = true;
        }
        let means = background.data().mean_axis(Axis(0)).unwrap();
        let scales = (0..background.n_features())
            .map(|feature| {
                let variance = background
                    .data()
                    .column(feature)
                    .iter()
                    .filter(|value| value.is_finite())
                    .map(|value| (value - means[feature]).powi(2))
                    .sum::<f64>()
                    / background.n_samples() as f64;
                let scale = variance.sqrt();
                if scale.is_finite() && scale > f64::EPSILON {
                    scale
                } else {
                    1.0
                }
            })
            .collect();
        Ok(Self {
            background,
            categorical,
            scales,
            neighbors,
        })
    }

    pub fn background(&self) -> &Background {
        &self.background
    }

    pub fn neighbors(&self) -> usize {
        self.neighbors
    }
}

impl Masker for ConditionalTabularMasker {
    fn n_features(&self) -> usize {
        self.background.n_features()
    }

    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        self.background.validate()?;
        if sample.len() != self.n_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let selected = if present.iter().any(|value| *value) {
            let mut distances = self
                .background
                .data()
                .rows()
                .into_iter()
                .enumerate()
                .map(|(row, values)| {
                    let distance = present
                        .iter()
                        .enumerate()
                        .filter(|(_, enabled)| **enabled)
                        .map(|(feature, _)| {
                            let left = sample[feature];
                            let right = values[feature];
                            if left.is_nan() || right.is_nan() {
                                if left.is_nan() && right.is_nan() {
                                    0.0
                                } else {
                                    1.0
                                }
                            } else if self.categorical[feature] {
                                if left == right {
                                    0.0
                                } else {
                                    1.0
                                }
                            } else {
                                ((left - right) / self.scales[feature]).powi(2)
                            }
                        })
                        .sum::<f64>();
                    (row, distance)
                })
                .collect::<Vec<_>>();
            distances.sort_by(|left, right| {
                left.1
                    .total_cmp(&right.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            distances
                .into_iter()
                .take(self.neighbors.min(self.background.n_samples()))
                .map(|(row, _)| row)
                .collect::<Vec<_>>()
        } else {
            (0..self.background.n_samples()).collect()
        };
        let mut output = self.background.select(&selected)?.data().to_owned();
        for (feature, enabled) in present.iter().copied().enumerate() {
            if enabled {
                output.column_mut(feature).fill(sample[feature]);
            }
        }
        Ok(output)
    }
}
#[cfg(test)]
mod structured_tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn text_masker_preserves_special_tokens() {
        let m = TextMasker::new(3, 99.)
            .unwrap()
            .with_immutable_positions(&[0])
            .unwrap();
        let out = m
            .mask(array![1., 2., 3.].view(), &[false, false, true])
            .unwrap();
        assert_eq!(out, array![[1., 99., 3.]]);
    }
    #[test]
    fn image_masker_uses_reference() {
        let m = ImageMasker::new(1, 1, 2, array![0.1, 0.2]).unwrap();
        let out = m.mask(array![0.8, 0.9].view(), &[true, false]).unwrap();
        assert_eq!(out, array![[0.8, 0.2]]);
    }
    #[test]
    fn conditional_tabular_masker_uses_categorical_and_numeric_distance() {
        let masker = ConditionalTabularMasker::new(
            Background::new(array![[0., 0.], [1., 0.1], [1., 10.], [2., 0.2]]).unwrap(),
            &[0],
            1,
        )
        .unwrap();
        let conditioned = masker.mask(array![1., 9.].view(), &[true, true]).unwrap();
        assert_eq!(conditioned, array![[1., 9.]]);
        let categorical_only = masker.mask(array![1., 9.].view(), &[true, false]).unwrap();
        assert_eq!(categorical_only, array![[1., 0.1]]);
        assert_eq!(
            masker
                .mask(array![1., 9.].view(), &[false, false])
                .unwrap()
                .nrows(),
            4
        );
    }
    #[test]
    fn grouped_masker_expands_coalitions_and_preserves_source_mapping() {
        let inner = FixedMasker::new(array![0., 0., 0.]).unwrap();
        let masker = GroupedMasker::new(inner, vec![vec![0, 2], vec![1]]).unwrap();
        let masked = masker
            .mask(array![2., 4., 8.].view(), &[true, false])
            .unwrap();
        assert_eq!(masked, array![[2., 0., 8.]]);
        assert_eq!(masker.groups(), &[vec![0, 2], vec![1]]);
        assert_eq!(
            masker
                .attribution_data(array![[2., 4., 8.]].view())
                .unwrap(),
            array![[5., 4.]]
        );
    }
    #[test]
    fn rejects_invalid_deserialized_style_maskers_before_indexing() {
        let text = TextMasker {
            n_tokens: 2,
            mask_token: 0.,
            immutable: vec![false],
        };
        assert!(text.mask(array![1., 2.].view(), &[false, false]).is_err());

        let image = ImageMasker {
            width: 2,
            height: 2,
            channels: 1,
            reference: array![0., 0.],
        };
        assert!(image.validate().is_err());
    }

    #[cfg(feature = "json-adapters")]
    #[test]
    fn serializable_builtin_maskers_round_trip() {
        let fixed = FixedMasker::new(array![0., 1.]).unwrap();
        let decoded: FixedMasker =
            serde_json::from_str(&serde_json::to_string(&fixed).unwrap()).unwrap();
        assert_eq!(decoded, fixed);

        let independent = IndependentMasker::new(Background::new(array![[0., 1.]]).unwrap());
        let decoded: IndependentMasker =
            serde_json::from_str(&serde_json::to_string(&independent).unwrap()).unwrap();
        assert_eq!(decoded, independent);
    }
}
