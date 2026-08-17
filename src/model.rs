//! Model prediction abstractions for `shap-rs`.
//!
//! The model layer intentionally knows nothing about SHAP itself.
//! It provides the interface through which explainers evaluate models.

use ndarray::{Array2, ArrayView2};

use crate::error::Result;

/// A model that can produce predictions for a batch of samples.
///
/// SHAP explainers frequently evaluate hundreds or thousands of masked
/// samples. Therefore the fundamental interface is batch-oriented rather
/// than sample-oriented.
///
/// Implementations should prefer efficient vectorized/batched prediction
/// whenever the underlying model supports it.
pub trait Predict {
    /// Predict outputs for a batch of samples.
    ///
    /// # Arguments
    ///
    /// * `x` - A two-dimensional array with shape
    ///   `(n_samples, n_features)`.
    ///
    /// # Returns
    ///
    /// A two-dimensional array with shape
    /// `(n_samples, n_outputs)`.
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>>;

    /// Returns the number of input features expected by the model.
    ///
    /// Implementations should return `None` when the model does not expose
    /// this information.
    fn n_features(&self) -> Option<usize> {
        None
    }

    /// Returns the number of outputs produced by the model.
    ///
    /// Implementations should return `None` when the model does not expose
    /// this information.
    fn n_outputs(&self) -> Option<usize> {
        None
    }
}

/// A convenience implementation allowing a prediction closure to be used
/// as a SHAP model.
///
/// This is particularly useful for model-agnostic explainers.
///
/// # Example
///
/// ```
/// use ndarray::{Array2, ArrayView2};
/// use shap_rs::model::Predict;
///
/// let model = FnModel::new(|x: ArrayView2<'_, f64>| {
///     let mut output = Array2::<f64>::zeros((x.nrows(), 1));
///
///     for i in 0..x.nrows() {
///         output[[i, 0]] = x[[i, 0]] + x[[i, 1]];
///     }
///
///     Ok(output)
/// });
/// ```
pub struct FnModel<F> {
    predict_fn: F,
    n_features: Option<usize>,
    n_outputs: Option<usize>,
}

impl<F> FnModel<F> {
    /// Creates a model from a batch prediction function.
    pub fn new(predict_fn: F) -> Self {
        Self {
            predict_fn,
            n_features: None,
            n_outputs: None,
        }
    }

    /// Sets the expected number of input features.
    pub fn with_n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    /// Sets the number of model outputs.
    pub fn with_n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = Some(n_outputs);
        self
    }

    /// Returns the configured feature count.
    pub fn configured_n_features(&self) -> Option<usize> {
        self.n_features
    }

    /// Returns the configured output count.
    pub fn configured_n_outputs(&self) -> Option<usize> {
        self.n_outputs
    }
}

impl<F> Predict for FnModel<F>
where
    F: Fn(ArrayView2<'_, f64>) -> Result<Array2<f64>>,
{
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        (self.predict_fn)(x)
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn n_outputs(&self) -> Option<usize> {
        self.n_outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fn_model_predicts_batch() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            let mut output = Array2::<f64>::zeros((x.nrows(), 1));

            for i in 0..x.nrows() {
                output[[i, 0]] = x[[i, 0]] + x[[i, 1]];
            }

            Ok(output)
        });

        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
        ];

        let predictions = model.predict(x.view()).unwrap();

        assert_eq!(predictions.shape(), &[2, 1]);
        assert_eq!(predictions[[0, 0]], 3.0);
        assert_eq!(predictions[[1, 0]], 7.0);
    }

    #[test]
    fn fn_model_can_store_dimensions() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(Array2::zeros((x.nrows(), 2)))
        })
        .with_n_features(4)
        .with_n_outputs(2);

        assert_eq!(model.n_features(), Some(4));
        assert_eq!(model.n_outputs(), Some(2));
    }

    #[test]
    fn fn_model_can_be_used_through_predict_trait() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(Array2::ones((x.nrows(), 1)))
        });

        let x = Array2::<f64>::zeros((3, 2));

        let predictions = Predict::predict(&model, x.view()).unwrap();

        assert_eq!(predictions.shape(), &[3, 1]);
        assert!(predictions.iter().all(|&value| value == 1.0));
    }
}