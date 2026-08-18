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

    /// Visits model-input batches for one coalition.
    ///
    /// The default implementation emits the result of [`Masker::mask`] once.
    /// Streaming maskers override this method so background rows can be read
    /// and evaluated incrementally without materializing the complete masked
    /// distribution.
    fn for_each_masked_batch(
        &self,
        sample: ArrayView1<'_, f64>,
        present: &[bool],
        visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>,
    ) -> Result<()> {
        visitor(self.mask(sample, present)?)
    }

    /// Whether [`Masker::for_each_masked_batch`] should be consumed directly
    /// instead of participating in ordinary multi-coalition batching.
    fn streams_masked_batches(&self) -> bool {
        false
    }
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
    fn for_each_masked_batch(
        &self,
        sample: ArrayView1<'_, f64>,
        present: &[bool],
        visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>,
    ) -> Result<()> {
        (**self).for_each_masked_batch(sample, present, visitor)
    }
    fn streams_masked_batches(&self) -> bool {
        (**self).streams_masked_batches()
    }
}

/// Adapts an incremental background source into a masker.
///
/// The callback may load rows from disk, a database, or another out-of-core
/// source and pass each masked batch to `visitor`. Every emitted batch must
/// have `n_features` columns and at least one row. Model-agnostic
/// explainers consume the batches immediately and retain only running output
/// sums. Calling [`Masker::mask`] directly remains supported but necessarily
/// collects all emitted batches.
pub struct FnStreamingMasker<F> {
    n_features: usize,
    batch_fn: F,
}

impl<F> FnStreamingMasker<F> {
    pub fn new(n_features: usize, batch_fn: F) -> Result<Self> {
        if n_features == 0 {
            return Err(ShapError::InvalidConfiguration(
                "streaming masker feature count must be positive".into(),
            ));
        }
        Ok(Self {
            n_features,
            batch_fn,
        })
    }
}

impl<F> Masker for FnStreamingMasker<F>
where
    F: Fn(ArrayView1<'_, f64>, &[bool], &mut dyn FnMut(Array2<f64>) -> Result<()>) -> Result<()>,
{
    fn n_features(&self) -> usize {
        self.n_features
    }

    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        let mut batches = Vec::new();
        self.for_each_masked_batch(sample, present, &mut |batch| {
            batches.push(batch);
            Ok(())
        })?;
        let rows = batches.iter().try_fold(0usize, |rows, batch| {
            rows.checked_add(batch.nrows()).ok_or_else(|| {
                ShapError::InvalidConfiguration("streaming masker row count overflow".into())
            })
        })?;
        crate::error::checked_f64_shape(
            &[rows, self.n_features],
            "collected streaming masker output",
        )?;
        let mut output = Array2::zeros((rows, self.n_features));
        let mut offset = 0;
        for batch in batches {
            let end = offset + batch.nrows();
            output
                .slice_axis_mut(Axis(0), ndarray::Slice::from(offset..end))
                .assign(&batch);
            offset = end;
        }
        Ok(output)
    }

    fn for_each_masked_batch(
        &self,
        sample: ArrayView1<'_, f64>,
        present: &[bool],
        visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>,
    ) -> Result<()> {
        if sample.len() != self.n_features || present.len() != self.n_features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut emitted = false;
        let mut checked_visitor = |batch: Array2<f64>| {
            if batch.nrows() == 0 || batch.ncols() != self.n_features {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("(rows>0, {}) streaming batch", self.n_features),
                    found: format!("{:?}", batch.dim()),
                });
            }
            emitted = true;
            visitor(batch)
        };
        (self.batch_fn)(sample, present, &mut checked_visitor)?;
        if !emitted {
            return Err(ShapError::MaskerError(
                "streaming masker returned no rows".into(),
            ));
        }
        Ok(())
    }

    fn streams_masked_batches(&self) -> bool {
        true
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

    fn for_each_masked_batch(
        &self,
        sample: ArrayView1<'_, f64>,
        present: &[bool],
        visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>,
    ) -> Result<()> {
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
        self.inner.for_each_masked_batch(sample, &expanded, visitor)
    }

    fn streams_masked_batches(&self) -> bool {
        self.inner.streams_masked_batches()
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

/// One tokenizer-produced piece used by [`TokenizedTextMasker`].
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "TextTokenPayload")]
pub struct TextToken {
    id: f64,
    text: String,
    special: bool,
    group: usize,
}

#[derive(serde::Deserialize)]
struct TextTokenPayload {
    id: f64,
    text: String,
    special: bool,
    group: usize,
}

impl TryFrom<TextTokenPayload> for TextToken {
    type Error = ShapError;

    fn try_from(payload: TextTokenPayload) -> Result<Self> {
        Ok(Self::new(payload.id, payload.text, payload.group)?.special(payload.special))
    }
}

impl TextToken {
    pub fn new(id: f64, text: impl Into<String>, group: usize) -> Result<Self> {
        if !id.is_finite() {
            return Err(ShapError::InvalidConfiguration(
                "text token ID must be finite".into(),
            ));
        }
        Ok(Self {
            id,
            text: text.into(),
            special: false,
            group,
        })
    }

    pub fn special(mut self, special: bool) -> Self {
        self.special = special;
        self
    }

    pub fn id(&self) -> f64 {
        self.id
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_special(&self) -> bool {
        self.special
    }

    pub fn group(&self) -> usize {
        self.group
    }
}

/// Controls whether tokenizer special pieces participate in coalitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SpecialTokenPolicy {
    /// Always retain special pieces and exclude them from the explanation.
    Preserve,
    /// Treat special pieces like ordinary pieces in their configured group.
    Mask,
}

/// Tokenizer-aware fixed-width text masking.
///
/// The caller supplies the tokenizer's pieces, numeric IDs, and grouping. This
/// keeps the crate tokenizer-independent while allowing reconstructed strings
/// to follow the tokenizer's exact piece spelling. Equal group labels share a
/// coalition bit, which is useful for word pieces or byte-pair tokens.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenizedTextMasker {
    tokens: Vec<TextToken>,
    position_groups: Vec<Option<usize>>,
    group_count: usize,
    mask_token_id: f64,
    mask_text: String,
    separator: String,
    special_policy: SpecialTokenPolicy,
}

impl TokenizedTextMasker {
    pub fn new(
        tokens: Vec<TextToken>,
        mask_token_id: f64,
        mask_text: impl Into<String>,
        special_policy: SpecialTokenPolicy,
    ) -> Result<Self> {
        if tokens.is_empty() || !mask_token_id.is_finite() {
            return Err(ShapError::InvalidConfiguration(
                "tokenized text masker requires tokens and a finite mask token ID".into(),
            ));
        }
        if tokens.iter().any(|token| !token.id.is_finite()) {
            return Err(ShapError::InvalidConfiguration(
                "text token IDs must be finite".into(),
            ));
        }
        let mut groups = std::collections::HashMap::new();
        let mut position_groups = Vec::with_capacity(tokens.len());
        for token in &tokens {
            if token.special && special_policy == SpecialTokenPolicy::Preserve {
                position_groups.push(None);
            } else {
                let next = groups.len();
                let group = *groups.entry(token.group).or_insert(next);
                position_groups.push(Some(group));
            }
        }
        if groups.is_empty() {
            return Err(ShapError::InvalidConfiguration(
                "tokenized text masker must expose at least one non-preserved token group".into(),
            ));
        }
        Ok(Self {
            tokens,
            position_groups,
            group_count: groups.len(),
            mask_token_id,
            mask_text: mask_text.into(),
            separator: String::new(),
            special_policy,
        })
    }

    /// Sets text inserted between reconstructed tokenizer pieces.
    pub fn with_separator(mut self, separator: impl Into<String>) -> Self {
        self.separator = separator.into();
        self
    }

    pub fn tokens(&self) -> &[TextToken] {
        &self.tokens
    }

    pub fn special_token_policy(&self) -> SpecialTokenPolicy {
        self.special_policy
    }

    /// Returns the model-ready token IDs represented by this tokenization.
    pub fn token_ids(&self) -> Array1<f64> {
        Array1::from_iter(self.tokens.iter().map(TextToken::id))
    }

    /// Reconstructs masked text using the tokenizer's original piece strings.
    pub fn reconstruct(&self, present: &[bool]) -> Result<String> {
        if present.len() != self.group_count {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} token groups", self.group_count),
                found: format!("{}", present.len()),
            });
        }
        Ok(self
            .tokens
            .iter()
            .zip(&self.position_groups)
            .map(|(token, group)| match group {
                None => token.text.as_str(),
                Some(group) if present[*group] => token.text.as_str(),
                Some(_) => self.mask_text.as_str(),
            })
            .collect::<Vec<_>>()
            .join(&self.separator))
    }
}

impl Masker for TokenizedTextMasker {
    fn n_features(&self) -> usize {
        self.group_count
    }

    fn n_input_features(&self) -> usize {
        self.tokens.len()
    }

    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if samples.ncols() != self.tokens.len() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} token IDs", self.tokens.len()),
                found: format!("{}", samples.ncols()),
            });
        }
        let mut grouped = Array2::zeros((samples.nrows(), self.group_count));
        let mut counts = vec![0usize; self.group_count];
        for group in self.position_groups.iter().flatten() {
            counts[*group] += 1;
        }
        for row in 0..samples.nrows() {
            for (position, group) in self.position_groups.iter().enumerate() {
                if let Some(group) = group {
                    grouped[[row, *group]] += samples[[row, position]];
                }
            }
            for group in 0..self.group_count {
                grouped[[row, group]] /= counts[group] as f64;
            }
        }
        Ok(grouped)
    }

    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        if sample.len() != self.tokens.len() || present.len() != self.group_count {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "{} token IDs and {} token groups",
                    self.tokens.len(),
                    self.group_count
                ),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let row = self
            .position_groups
            .iter()
            .enumerate()
            .map(|(position, group)| match group {
                None => sample[position],
                Some(group) if present[*group] => sample[position],
                Some(_) => self.mask_token_id,
            })
            .collect::<Vec<_>>();
        Array2::from_shape_vec((1, row.len()), row)
            .map_err(|error| ShapError::MaskerError(error.to_string()))
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

/// Baseline used to replace absent image segments.
#[derive(Debug, Clone, PartialEq)]
pub enum ImageBaseline {
    /// A fixed flattened image with the same dimensions as the input.
    Reference(Array1<f64>),
    /// A box blur computed from each input image. `radius` is measured in pixels.
    Blur { radius: usize },
}

/// Masks a flattened image by segment (for example, by superpixel).
/// All channels belonging to a pixel are controlled by the same coalition bit.
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentedImageMasker {
    width: usize,
    height: usize,
    channels: usize,
    segments: Vec<usize>,
    segment_count: usize,
    baseline: ImageBaseline,
}

impl SegmentedImageMasker {
    pub fn new(
        width: usize,
        height: usize,
        channels: usize,
        segments: Vec<usize>,
        baseline: ImageBaseline,
    ) -> Result<Self> {
        let pixels = width
            .checked_mul(height)
            .ok_or_else(|| ShapError::InvalidConfiguration("image dimensions overflow".into()))?;
        let values = pixels
            .checked_mul(channels)
            .ok_or_else(|| ShapError::InvalidConfiguration("image dimensions overflow".into()))?;
        if values == 0 || segments.len() != pixels {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{pixels} segment labels"),
                found: format!("{}", segments.len()),
            });
        }
        let mut labels = std::collections::HashMap::new();
        let mut normalized = Vec::with_capacity(pixels);
        for label in segments {
            let next = labels.len();
            let index = *labels.entry(label).or_insert(next);
            normalized.push(index);
        }
        match &baseline {
            ImageBaseline::Reference(reference) if reference.len() != values => {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{values} reference values"),
                    found: format!("{}", reference.len()),
                });
            }
            ImageBaseline::Blur { radius: 0 } => {
                return Err(ShapError::InvalidConfiguration(
                    "image blur radius must be positive".into(),
                ));
            }
            _ => {}
        }
        Ok(Self {
            width,
            height,
            channels,
            segment_count: labels.len(),
            segments: normalized,
            baseline,
        })
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.channels)
    }

    pub fn segments(&self) -> &[usize] {
        &self.segments
    }

    pub fn baseline(&self) -> &ImageBaseline {
        &self.baseline
    }

    fn blurred(&self, sample: ArrayView1<'_, f64>, radius: usize) -> Array1<f64> {
        let mut output = Array1::zeros(sample.len());
        for y in 0..self.height {
            let y0 = y.saturating_sub(radius);
            let y1 = y.saturating_add(radius).min(self.height - 1);
            for x in 0..self.width {
                let x0 = x.saturating_sub(radius);
                let x1 = x.saturating_add(radius).min(self.width - 1);
                let count = (y1 - y0 + 1) * (x1 - x0 + 1);
                for channel in 0..self.channels {
                    let sum = (y0..=y1)
                        .flat_map(|source_y| (x0..=x1).map(move |source_x| (source_y, source_x)))
                        .map(|(source_y, source_x)| {
                            sample[(source_y * self.width + source_x) * self.channels + channel]
                        })
                        .sum::<f64>();
                    output[(y * self.width + x) * self.channels + channel] = sum / count as f64;
                }
            }
        }
        output
    }
}

impl Masker for SegmentedImageMasker {
    fn n_features(&self) -> usize {
        self.segment_count
    }
    fn n_input_features(&self) -> usize {
        self.width * self.height * self.channels
    }

    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if samples.ncols() != self.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} image values", self.n_input_features()),
                found: format!("{}", samples.ncols()),
            });
        }
        let mut grouped = Array2::zeros((samples.nrows(), self.segment_count));
        let mut counts = vec![0usize; self.segment_count];
        for &segment in &self.segments {
            counts[segment] += self.channels;
        }
        for row in 0..samples.nrows() {
            for (pixel, &segment) in self.segments.iter().enumerate() {
                for channel in 0..self.channels {
                    grouped[[row, segment]] += samples[[row, pixel * self.channels + channel]];
                }
            }
            for segment in 0..self.segment_count {
                grouped[[row, segment]] /= counts[segment] as f64;
            }
        }
        Ok(grouped)
    }

    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        if sample.len() != self.n_input_features() || present.len() != self.segment_count {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "{} image values and {} segments",
                    self.n_input_features(),
                    self.segment_count
                ),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let mut output = match &self.baseline {
            ImageBaseline::Reference(reference) => reference.clone(),
            ImageBaseline::Blur { radius } => self.blurred(sample, *radius),
        };
        for (pixel, &segment) in self.segments.iter().enumerate() {
            if present[segment] {
                for channel in 0..self.channels {
                    let index = pixel * self.channels + channel;
                    output[index] = sample[index];
                }
            }
        }
        Array2::from_shape_vec((1, output.len()), output.to_vec())
            .map_err(|error| ShapError::MaskerError(error.to_string()))
    }
}

/// Adapts an image inpainting function into a segment-aware masker.
pub struct InpaintingImageMasker<F> {
    segmented: SegmentedImageMasker,
    inpaint: F,
}

impl<F> InpaintingImageMasker<F> {
    pub fn new(segmented: SegmentedImageMasker, inpaint: F) -> Self {
        Self { segmented, inpaint }
    }
    pub fn segmented(&self) -> &SegmentedImageMasker {
        &self.segmented
    }
}

impl<F> Masker for InpaintingImageMasker<F>
where
    F: Fn(ArrayView1<'_, f64>, &[bool], &[usize], (usize, usize, usize)) -> Result<Array2<f64>>,
{
    fn n_features(&self) -> usize {
        self.segmented.n_features()
    }
    fn n_input_features(&self) -> usize {
        self.segmented.n_input_features()
    }
    fn attribution_data(&self, samples: ndarray::ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.segmented.attribution_data(samples)
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        if sample.len() != self.n_input_features() || present.len() != self.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "{} image values and {} segments",
                    self.n_input_features(),
                    self.n_features()
                ),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let output = (self.inpaint)(
            sample,
            present,
            self.segmented.segments(),
            self.segmented.dimensions(),
        )?;
        if output.nrows() == 0 || output.ncols() != self.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "(rows>0, {}) inpainted image batch",
                    self.n_input_features()
                ),
                found: format!("{:?}", output.dim()),
            });
        }
        Ok(output)
    }
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
    fn tokenized_text_groups_pieces_and_reconstructs_text() {
        let masker = TokenizedTextMasker::new(
            vec![
                TextToken::new(101., "[CLS]", 99).unwrap().special(true),
                TextToken::new(10., "walk", 0).unwrap(),
                TextToken::new(11., "##ing", 0).unwrap(),
                TextToken::new(20., " home", 1).unwrap(),
                TextToken::new(102., "[SEP]", 100).unwrap().special(true),
            ],
            0.,
            "[MASK]",
            SpecialTokenPolicy::Preserve,
        )
        .unwrap();
        assert_eq!(masker.n_features(), 2);
        assert_eq!(masker.n_input_features(), 5);
        assert_eq!(
            masker
                .mask(masker.token_ids().view(), &[false, true])
                .unwrap(),
            array![[101., 0., 0., 20., 102.]]
        );
        assert_eq!(
            masker.reconstruct(&[false, true]).unwrap(),
            "[CLS][MASK][MASK] home[SEP]"
        );
        assert_eq!(
            masker
                .attribution_data(array![[101., 10., 12., 20., 102.]].view())
                .unwrap(),
            array![[11., 20.]]
        );
    }

    #[test]
    fn tokenized_text_can_mask_special_tokens() {
        let masker = TokenizedTextMasker::new(
            vec![
                TextToken::new(101., "CLS", 0).unwrap().special(true),
                TextToken::new(5., "word", 1).unwrap(),
            ],
            -1.,
            "_",
            SpecialTokenPolicy::Mask,
        )
        .unwrap()
        .with_separator(" ");
        assert_eq!(masker.reconstruct(&[false, true]).unwrap(), "_ word");
        assert_eq!(
            masker
                .mask(masker.token_ids().view(), &[false, true])
                .unwrap(),
            array![[-1., 5.]]
        );
    }
    #[test]
    fn image_masker_uses_reference() {
        let m = ImageMasker::new(1, 1, 2, array![0.1, 0.2]).unwrap();
        let out = m.mask(array![0.8, 0.9].view(), &[true, false]).unwrap();
        assert_eq!(out, array![[0.8, 0.2]]);
    }
    #[test]
    fn segmented_image_masker_controls_pixels_and_channels_together() {
        let masker = SegmentedImageMasker::new(
            2,
            1,
            2,
            vec![7, 9],
            ImageBaseline::Reference(array![0.1, 0.2, 0.3, 0.4]),
        )
        .unwrap();
        let output = masker
            .mask(array![1., 2., 3., 4.].view(), &[true, false])
            .unwrap();
        assert_eq!(output, array![[1., 2., 0.3, 0.4]]);
        assert_eq!(masker.n_features(), 2);
        assert_eq!(
            masker
                .attribution_data(array![[1., 3., 5., 7.]].view())
                .unwrap(),
            array![[2., 6.]]
        );
    }

    #[test]
    fn segmented_image_blur_is_channel_aware() {
        let masker =
            SegmentedImageMasker::new(3, 1, 1, vec![0, 1, 2], ImageBaseline::Blur { radius: 1 })
                .unwrap();
        let output = masker
            .mask(array![0., 3., 9.].view(), &[false, false, false])
            .unwrap();
        assert_eq!(output, array![[1.5, 4., 6.]]);
    }

    #[test]
    fn inpainting_adapter_validates_callback_shape() {
        let segmented =
            SegmentedImageMasker::new(1, 1, 1, vec![0], ImageBaseline::Reference(array![0.]))
                .unwrap();
        let masker = InpaintingImageMasker::new(
            segmented,
            |_: ArrayView1<'_, f64>,
             _: &[bool],
             _: &[usize],
             _: (usize, usize, usize)|
             -> Result<Array2<f64>> { Ok(Array2::zeros((1, 0))) },
        );
        assert!(masker.mask(array![1.].view(), &[false]).is_err());
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
