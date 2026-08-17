//! Background datasets used by SHAP explainers.
//!
//! A background dataset represents the reference distribution against which
//! feature contributions are measured. For tabular SHAP explainers, this is
//! typically a matrix of observations with shape `(n_samples, n_features)`.

use ndarray::{Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::Rng;

use crate::error::{Result, ShapError};

/// A reference dataset used by a SHAP explainer.
///
/// `Background` owns its data so that explainers can safely retain it for
/// their entire lifetime.
///
/// The data has shape:
///
/// ```text
/// (n_background_samples, n_features)
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Background {
    data: Array2<f64>,
}

impl Background {
    /// Revalidates a background after deserialization.
    pub fn validate(&self) -> Result<()> {
        if self.data.nrows() == 0 {
            return Err(ShapError::EmptyBackground);
        }
        if self.data.ncols() == 0 {
            return Err(ShapError::InvalidConfiguration(
                "background must contain at least one feature".into(),
            ));
        }
        Ok(())
    }
    /// Creates a background dataset from an owned array.
    ///
    /// # Errors
    ///
    /// Returns [`ShapError::EmptyBackground`] when the supplied array has
    /// zero rows.
    pub fn new(data: Array2<f64>) -> Result<Self> {
        if data.nrows() == 0 {
            return Err(ShapError::EmptyBackground);
        }
        if data.ncols() == 0 {
            return Err(ShapError::InvalidConfiguration(
                "background must contain at least one feature".to_string(),
            ));
        }

        Ok(Self { data })
    }

    /// Creates a background dataset by copying an array view.
    pub fn from_view(data: ArrayView2<'_, f64>) -> Result<Self> {
        Self::new(data.to_owned())
    }

    /// Returns the underlying background data.
    pub fn data(&self) -> ArrayView2<'_, f64> {
        self.data.view()
    }

    /// Returns the number of background observations.
    pub fn n_samples(&self) -> usize {
        self.data.nrows()
    }

    /// Returns the number of features.
    pub fn n_features(&self) -> usize {
        self.data.ncols()
    }

    /// Returns one background observation.
    ///
    /// # Errors
    ///
    /// Returns [`ShapError::InvalidConfiguration`] if `index` is outside
    /// the background dataset.
    pub fn row(&self, index: usize) -> Result<ArrayView1<'_, f64>> {
        self.validate()?;
        if index >= self.n_samples() {
            return Err(ShapError::InvalidConfiguration(format!(
                "background row index {index} is out of bounds for {} rows",
                self.n_samples()
            )));
        }
        self.data
            .row(index)
            .into_shape_with_order(self.n_features())
            .map_err(|_| {
                ShapError::InvalidConfiguration(format!("unable to access background row {index}"))
            })
    }

    /// Creates a new background dataset containing a random sample of the
    /// existing observations.
    ///
    /// Sampling is performed without replacement.
    ///
    /// If `n_samples` is greater than or equal to the current number of
    /// observations, a clone of the complete background dataset is returned.
    pub fn sample<R: Rng + ?Sized>(&self, n_samples: usize, rng: &mut R) -> Result<Self> {
        self.validate()?;
        if n_samples == 0 {
            return Err(ShapError::InvalidConfiguration(
                "background sample size must be greater than zero".to_string(),
            ));
        }

        if n_samples >= self.n_samples() {
            return Ok(self.clone());
        }

        let mut indices: Vec<usize> = (0..self.n_samples()).collect();
        indices.shuffle(rng);
        indices.truncate(n_samples);

        let sampled = self.data.select(ndarray::Axis(0), &indices);

        Self::new(sampled)
    }

    /// Returns a background dataset containing the specified rows.
    ///
    /// This method preserves the order of `indices`.
    pub fn select(&self, indices: &[usize]) -> Result<Self> {
        self.validate()?;
        if indices.is_empty() {
            return Err(ShapError::EmptyBackground);
        }

        for &index in indices {
            if index >= self.n_samples() {
                return Err(ShapError::InvalidConfiguration(format!(
                    "background row index {index} is out of bounds for {} rows",
                    self.n_samples()
                )));
            }
        }

        Self::new(self.data.select(ndarray::Axis(0), indices))
    }
}

/// Strategy used to reduce a large background dataset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
pub enum BackgroundSampling {
    /// Use every background observation.
    #[default]
    All,

    /// Randomly sample a fixed number of observations.
    Random(usize),
}

impl BackgroundSampling {
    /// Applies this sampling strategy to a background dataset.
    pub fn apply<R: Rng + ?Sized>(
        self,
        background: &Background,
        rng: &mut R,
    ) -> Result<Background> {
        match self {
            Self::All => Ok(background.clone()),
            Self::Random(n_samples) => background.sample(n_samples, rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn creates_background() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let background = Background::new(data).unwrap();

        assert_eq!(background.n_samples(), 3);
        assert_eq!(background.n_features(), 2);
    }

    #[test]
    fn rejects_empty_background() {
        let data = Array2::<f64>::zeros((0, 2));

        let result = Background::new(data);

        assert!(matches!(result, Err(ShapError::EmptyBackground)));
    }

    #[test]
    fn exposes_data() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data.clone()).unwrap();

        assert_eq!(background.data(), data.view());
    }

    #[test]
    fn accesses_row() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data).unwrap();

        let row = background.row(1).unwrap();

        assert_eq!(row, array![3.0, 4.0].view());
    }

    #[test]
    fn random_sampling_is_reproducible() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0],];

        let background = Background::new(data).unwrap();

        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);

        let sample1 = background.sample(3, &mut rng1).unwrap();
        let sample2 = background.sample(3, &mut rng2).unwrap();

        assert_eq!(sample1.data(), sample2.data());
    }

    #[test]
    fn sampling_all_rows_returns_clone() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data.clone()).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        let sampled = background.sample(10, &mut rng).unwrap();

        assert_eq!(sampled.data(), data.view());
    }

    #[test]
    fn selects_rows() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let background = Background::new(data).unwrap();

        let selected = background.select(&[2, 0]).unwrap();

        assert_eq!(selected.data(), array![[5.0, 6.0], [1.0, 2.0],].view());
    }

    #[test]
    fn rejects_empty_selection() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data).unwrap();

        let result = background.select(&[]);

        assert!(matches!(result, Err(ShapError::EmptyBackground)));
    }

    #[test]
    fn rejects_invalid_selection_index() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data).unwrap();

        let result = background.select(&[5]);

        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }

    #[test]
    fn sampling_strategy_all() {
        let data = array![[1.0, 2.0], [3.0, 4.0],];

        let background = Background::new(data.clone()).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        let result = BackgroundSampling::All
            .apply(&background, &mut rng)
            .unwrap();

        assert_eq!(result.data(), data.view());
    }

    #[test]
    fn sampling_strategy_random() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0],];

        let background = Background::new(data).unwrap();

        let mut rng = StdRng::seed_from_u64(42);

        let result = BackgroundSampling::Random(2)
            .apply(&background, &mut rng)
            .unwrap();

        assert_eq!(result.n_samples(), 2);
        assert_eq!(result.n_features(), 2);
    }
}
