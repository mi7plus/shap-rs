use crate::{Background, Result, ShapError};
use ndarray::{Array1, Array2, ArrayView1, Axis};
pub trait Masker {
    fn n_features(&self) -> usize;
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>>;
}
impl<T: Masker + ?Sized> Masker for &T {
    fn n_features(&self) -> usize {
        (**self).n_features()
    }
    fn mask(&self, sample: ArrayView1<'_, f64>, present: &[bool]) -> Result<Array2<f64>> {
        (**self).mask(sample, present)
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
pub struct FixedMasker {
    reference: Array1<f64>,
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
pub struct TextMasker {
    n_tokens: usize,
    mask_token: f64,
    immutable: Vec<bool>,
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
pub struct ImageMasker {
    width: usize,
    height: usize,
    channels: usize,
    reference: Array1<f64>,
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
