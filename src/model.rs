//! Model prediction abstractions for `shap-rs`.
//!
//! The model layer intentionally knows nothing about SHAP itself.
//! It provides the interface through which explainers evaluate models.

use ndarray::{Array2, Array3, ArrayView2};
use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

use crate::error::{Result, ShapError};

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

    /// Predicts from an owned batch.
    ///
    /// The default delegates to [`Predict::predict`]. Device adapters may
    /// override this to move a coalition batch directly into their transfer
    /// or tensor-construction path without an intermediate host copy.
    fn predict_owned(&self, x: Array2<f64>) -> Result<Array2<f64>> {
        self.predict(x.view())
    }

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
impl<T: Predict + ?Sized> Predict for &T {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        (**self).predict(x)
    }
    fn predict_owned(&self, x: Array2<f64>) -> Result<Array2<f64>> {
        (**self).predict_owned(x)
    }
    fn n_features(&self) -> Option<usize> {
        (**self).n_features()
    }
    fn n_outputs(&self) -> Option<usize> {
        (**self).n_outputs()
    }
}

#[derive(Debug)]
struct PredictionCacheState {
    rows: HashMap<Vec<u64>, Vec<f64>>,
    order: VecDeque<Vec<u64>>,
    outputs: Option<usize>,
}

/// Adds bounded, reusable row-level prediction caching to a deterministic model.
///
/// Cache keys use every input value's exact IEEE-754 bits, so distinct NaN
/// payloads and signed zero are not conflated. The wrapper is opt-in because it
/// is only semantically safe for models whose predictions do not change for an
/// identical row. Its cache is shared across samples, explainer calls, and
/// threads; concurrent misses may be evaluated more than once but never corrupt
/// cached values.
pub struct CachedModel<M> {
    inner: M,
    capacity: usize,
    state: Mutex<PredictionCacheState>,
}

impl<M> CachedModel<M> {
    pub fn new(inner: M, capacity: usize) -> Result<Self> {
        if capacity == 0 {
            return Err(ShapError::InvalidConfiguration(
                "prediction cache capacity must be positive".into(),
            ));
        }
        Ok(Self {
            inner,
            capacity,
            state: Mutex::new(PredictionCacheState {
                rows: HashMap::new(),
                order: VecDeque::new(),
                outputs: None,
            }),
        })
    }

    pub fn inner(&self) -> &M {
        &self.inner
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> Result<usize> {
        Ok(self.lock_state()?.rows.len())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    pub fn clear(&self) -> Result<()> {
        let mut state = self.lock_state()?;
        state.rows.clear();
        state.order.clear();
        state.outputs = None;
        Ok(())
    }

    fn lock_state(&self) -> Result<std::sync::MutexGuard<'_, PredictionCacheState>> {
        self.state
            .lock()
            .map_err(|_| ShapError::Other("prediction cache lock was poisoned".into()))
    }

    fn touch(state: &mut PredictionCacheState, key: &[u64]) {
        if let Some(position) = state.order.iter().position(|cached| cached == key) {
            state.order.remove(position);
        }
        state.order.push_back(key.to_vec());
    }
}

impl<M: Predict> Predict for CachedModel<M> {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if let Some(features) = self.inner.n_features() {
            if features != x.ncols() {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("{features} model features"),
                    found: format!("{}", x.ncols()),
                });
            }
        }
        if x.nrows() == 0 {
            let predictions = self.inner.predict(x)?;
            if predictions.nrows() != 0 || predictions.ncols() == 0 {
                return Err(ShapError::DimensionMismatch {
                    expected: "(0, outputs>0)".into(),
                    found: format!("{:?}", predictions.dim()),
                });
            }
            let mut state = self.lock_state()?;
            if state
                .outputs
                .is_some_and(|outputs| outputs != predictions.ncols())
            {
                return Err(ShapError::OutputDimensionMismatch {
                    expected: state.outputs.unwrap(),
                    found: predictions.ncols(),
                });
            }
            state.outputs = Some(predictions.ncols());
            return Ok(predictions);
        }
        let keys = x
            .rows()
            .into_iter()
            .map(|row| row.iter().map(|value| value.to_bits()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let mut values = vec![None; x.nrows()];
        let mut missing = Vec::<Vec<u64>>::new();
        let mut missing_lookup = HashMap::<Vec<u64>, usize>::new();
        {
            let mut state = self.lock_state()?;
            for (row, key) in keys.iter().enumerate() {
                if let Some(cached) = state.rows.get(key).cloned() {
                    Self::touch(&mut state, key);
                    values[row] = Some(cached);
                } else if !missing_lookup.contains_key(key) {
                    missing_lookup.insert(key.clone(), missing.len());
                    missing.push(key.clone());
                }
            }
        }

        let mut predicted_missing = Vec::new();
        if !missing.is_empty() {
            crate::error::checked_f64_shape(
                &[missing.len(), x.ncols()],
                "prediction cache miss batch",
            )?;
            let mut batch = Array2::zeros((missing.len(), x.ncols()));
            for (row, key) in missing.iter().enumerate() {
                for (column, bits) in key.iter().enumerate() {
                    batch[[row, column]] = f64::from_bits(*bits);
                }
            }
            let predictions = self.inner.predict_owned(batch)?;
            if predictions.nrows() != missing.len() || predictions.ncols() == 0 {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("({}, outputs>0)", missing.len()),
                    found: format!("{:?}", predictions.dim()),
                });
            }
            if predictions.iter().any(|value| !value.is_finite()) {
                return Err(ShapError::ModelError(
                    "prediction contains a non-finite value".into(),
                ));
            }
            predicted_missing = predictions
                .rows()
                .into_iter()
                .map(|row| row.to_vec())
                .collect();
            let mut state = self.lock_state()?;
            if state
                .outputs
                .is_some_and(|outputs| outputs != predictions.ncols())
            {
                return Err(ShapError::OutputDimensionMismatch {
                    expected: state.outputs.unwrap(),
                    found: predictions.ncols(),
                });
            }
            state.outputs = Some(predictions.ncols());
            for (key, prediction) in missing.iter().zip(&predicted_missing) {
                while state.rows.len() >= self.capacity {
                    let Some(evicted) = state.order.pop_front() else {
                        break;
                    };
                    state.rows.remove(&evicted);
                }
                state.rows.insert(key.clone(), prediction.clone());
                Self::touch(&mut state, key);
            }
        }

        for (row, key) in keys.iter().enumerate() {
            if values[row].is_none() {
                values[row] = Some(predicted_missing[missing_lookup[key]].clone());
            }
        }
        let outputs = values
            .first()
            .and_then(Option::as_ref)
            .map(Vec::len)
            .or_else(|| self.lock_state().ok().and_then(|state| state.outputs))
            .or_else(|| self.inner.n_outputs())
            .unwrap_or(0);
        let flat = values.into_iter().flatten().flatten().collect::<Vec<_>>();
        Array2::from_shape_vec((x.nrows(), outputs), flat)
            .map_err(|error| ShapError::ModelError(error.to_string()))
    }

    fn n_features(&self) -> Option<usize> {
        self.inner.n_features()
    }

    fn n_outputs(&self) -> Option<usize> {
        self.inner
            .n_outputs()
            .or_else(|| self.lock_state().ok().and_then(|state| state.outputs))
    }
}
/// A prediction model exposing input gradients with shape
/// `(samples, features, outputs)`.
pub trait DifferentiablePredict: Predict {
    fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>>;
}
impl<T: DifferentiablePredict + ?Sized> DifferentiablePredict for &T {
    fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>> {
        (**self).gradients(x)
    }
}
/// Neural-network adapter capable of propagating Deep SHAP multipliers.
pub trait DeepAttribution: Predict {
    fn deep_contributions(
        &self,
        x: ArrayView2<'_, f64>,
        background: ArrayView2<'_, f64>,
    ) -> Result<Array3<f64>>;
}
impl<T: DeepAttribution + ?Sized> DeepAttribution for &T {
    fn deep_contributions(
        &self,
        x: ArrayView2<'_, f64>,
        background: ArrayView2<'_, f64>,
    ) -> Result<Array3<f64>> {
        (**self).deep_contributions(x, background)
    }
}
/// Execution target exposed by accelerated model adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExecutionDevice {
    Cpu,
    Cuda(u16),
    Metal,
    Vulkan,
    WebGpu,
}
pub trait AcceleratedPredict: Predict {
    fn predict_on(&self, x: ArrayView2<'_, f64>, device: ExecutionDevice) -> Result<Array2<f64>>;

    /// Owned-batch device prediction fast path.
    fn predict_owned_on(&self, x: Array2<f64>, device: ExecutionDevice) -> Result<Array2<f64>> {
        self.predict_on(x.view(), device)
    }
}
/// Adapts a device-aware prediction closure into an accelerated model.
///
/// Ordinary [`Predict`] calls use [`ExecutionDevice::Cpu`]; bind another
/// device with [`DeviceModel`] when passing the model to an explainer.
pub struct FnAcceleratedModel<F> {
    predict_fn: F,
    n_features: Option<usize>,
    n_outputs: Option<usize>,
}
impl<F> FnAcceleratedModel<F> {
    pub fn new(predict_fn: F) -> Self {
        Self {
            predict_fn,
            n_features: None,
            n_outputs: None,
        }
    }
    pub fn with_n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }
    pub fn with_n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = Some(n_outputs);
        self
    }
}
impl<F> Predict for FnAcceleratedModel<F>
where
    F: Fn(ArrayView2<'_, f64>, ExecutionDevice) -> Result<Array2<f64>>,
{
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        (self.predict_fn)(x, ExecutionDevice::Cpu)
    }
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
    fn n_outputs(&self) -> Option<usize> {
        self.n_outputs
    }
}
impl<F> AcceleratedPredict for FnAcceleratedModel<F>
where
    F: Fn(ArrayView2<'_, f64>, ExecutionDevice) -> Result<Array2<f64>>,
{
    fn predict_on(&self, x: ArrayView2<'_, f64>, device: ExecutionDevice) -> Result<Array2<f64>> {
        (self.predict_fn)(x, device)
    }
}

/// Adapts an owned-batch device prediction closure.
///
/// This is the preferred closure adapter for GPU coalition evaluation because
/// the evaluator's combined batch moves into the closure exactly once.
pub struct FnOwnedAcceleratedModel<F> {
    predict_fn: F,
    n_features: Option<usize>,
    n_outputs: Option<usize>,
}

impl<F> FnOwnedAcceleratedModel<F> {
    pub fn new(predict_fn: F) -> Self {
        Self {
            predict_fn,
            n_features: None,
            n_outputs: None,
        }
    }

    pub fn with_n_features(mut self, n_features: usize) -> Self {
        self.n_features = Some(n_features);
        self
    }

    pub fn with_n_outputs(mut self, n_outputs: usize) -> Self {
        self.n_outputs = Some(n_outputs);
        self
    }
}

impl<F> Predict for FnOwnedAcceleratedModel<F>
where
    F: Fn(Array2<f64>, ExecutionDevice) -> Result<Array2<f64>>,
{
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        (self.predict_fn)(x.to_owned(), ExecutionDevice::Cpu)
    }

    fn predict_owned(&self, x: Array2<f64>) -> Result<Array2<f64>> {
        (self.predict_fn)(x, ExecutionDevice::Cpu)
    }

    fn n_features(&self) -> Option<usize> {
        self.n_features
    }

    fn n_outputs(&self) -> Option<usize> {
        self.n_outputs
    }
}

impl<F> AcceleratedPredict for FnOwnedAcceleratedModel<F>
where
    F: Fn(Array2<f64>, ExecutionDevice) -> Result<Array2<f64>>,
{
    fn predict_on(&self, x: ArrayView2<'_, f64>, device: ExecutionDevice) -> Result<Array2<f64>> {
        (self.predict_fn)(x.to_owned(), device)
    }

    fn predict_owned_on(&self, x: Array2<f64>, device: ExecutionDevice) -> Result<Array2<f64>> {
        (self.predict_fn)(x, device)
    }
}
/// Binds an accelerated model to a device while retaining the ordinary
/// [`Predict`] interface consumed by every explainer.
pub struct DeviceModel<'a, M> {
    model: &'a M,
    device: ExecutionDevice,
}
impl<'a, M> DeviceModel<'a, M> {
    pub fn new(model: &'a M, device: ExecutionDevice) -> Self {
        Self { model, device }
    }
    pub fn device(&self) -> ExecutionDevice {
        self.device
    }
}
impl<M: AcceleratedPredict> Predict for DeviceModel<'_, M> {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.model.predict_on(x, self.device)
    }
    fn predict_owned(&self, x: Array2<f64>) -> Result<Array2<f64>> {
        self.model.predict_owned_on(x, self.device)
    }
    fn n_features(&self) -> Option<usize> {
        self.model.n_features()
    }
    fn n_outputs(&self) -> Option<usize> {
        self.model.n_outputs()
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
/// use shap_rs::model::{FnModel, Predict};
///
/// let model = FnModel::new(|x: ArrayView2<'_, f64>| -> shap_rs::Result<Array2<f64>> {
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
    use std::cell::Cell;

    #[test]
    fn fn_model_predicts_batch() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            let mut output = Array2::<f64>::zeros((x.nrows(), 1));

            for i in 0..x.nrows() {
                output[[i, 0]] = x[[i, 0]] + x[[i, 1]];
            }

            Ok(output)
        });

        let x = array![[1.0, 2.0], [3.0, 4.0],];

        let predictions = model.predict(x.view()).unwrap();

        assert_eq!(predictions.shape(), &[2, 1]);
        assert_eq!(predictions[[0, 0]], 3.0);
        assert_eq!(predictions[[1, 0]], 7.0);
    }

    #[test]
    fn accelerated_closure_receives_bound_device() {
        let model = FnAcceleratedModel::new(|x: ArrayView2<'_, f64>, device| {
            let offset = if device == ExecutionDevice::Cuda(2) {
                10.0
            } else {
                0.0
            };
            Ok(Array2::from_shape_fn((x.nrows(), 1), |(i, _)| {
                x[[i, 0]] + offset
            }))
        });
        let bound = DeviceModel::new(&model, ExecutionDevice::Cuda(2));
        assert_eq!(bound.predict(array![[3.0]].view()).unwrap()[[0, 0]], 13.0);
        assert_eq!(model.predict(array![[3.0]].view()).unwrap()[[0, 0]], 3.0);
    }

    #[test]
    fn device_model_dispatches_owned_batches_to_transfer_fast_path() {
        struct TrackingModel {
            borrowed_calls: Cell<usize>,
            owned_calls: Cell<usize>,
        }
        impl Predict for TrackingModel {
            fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
                Ok(x.to_owned())
            }
        }
        impl AcceleratedPredict for TrackingModel {
            fn predict_on(
                &self,
                x: ArrayView2<'_, f64>,
                _: ExecutionDevice,
            ) -> Result<Array2<f64>> {
                self.borrowed_calls.set(self.borrowed_calls.get() + 1);
                Ok(x.to_owned())
            }
            fn predict_owned_on(&self, x: Array2<f64>, _: ExecutionDevice) -> Result<Array2<f64>> {
                self.owned_calls.set(self.owned_calls.get() + 1);
                Ok(x)
            }
        }
        let model = TrackingModel {
            borrowed_calls: Cell::new(0),
            owned_calls: Cell::new(0),
        };
        let bound = DeviceModel::new(&model, ExecutionDevice::Cuda(0));
        let output = bound.predict_owned(array![[1., 2.]]).unwrap();
        assert_eq!(output, array![[1., 2.]]);
        assert_eq!(model.owned_calls.get(), 1);
        assert_eq!(model.borrowed_calls.get(), 0);
    }

    #[test]
    fn owned_accelerated_closure_receives_batch_and_device() {
        let model = FnOwnedAcceleratedModel::new(|x: Array2<f64>, device| {
            let offset = if device == ExecutionDevice::Vulkan {
                2.0
            } else {
                0.0
            };
            Ok(x + offset)
        });
        let bound = DeviceModel::new(&model, ExecutionDevice::Vulkan);
        assert_eq!(
            bound.predict_owned(array![[1., 3.]]).unwrap(),
            array![[3., 5.]]
        );
    }

    #[test]
    fn fn_model_can_store_dimensions() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(Array2::zeros((x.nrows(), 2))))
            .with_n_features(4)
            .with_n_outputs(2);

        assert_eq!(model.n_features(), Some(4));
        assert_eq!(model.n_outputs(), Some(2));
    }

    #[test]
    fn fn_model_can_be_used_through_predict_trait() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(Array2::ones((x.nrows(), 1))));

        let x = Array2::<f64>::zeros((3, 2));

        let predictions = Predict::predict(&model, x.view()).unwrap();

        assert_eq!(predictions.shape(), &[3, 1]);
        assert!(predictions.iter().all(|&value| value == 1.0));
    }

    #[test]
    fn cached_model_reuses_rows_across_batches_and_deduplicates_misses() {
        let calls = Cell::new(0usize);
        let rows = Cell::new(0usize);
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            calls.set(calls.get() + 1);
            rows.set(rows.get() + x.nrows());
            Ok(x.sum_axis(ndarray::Axis(1)).insert_axis(ndarray::Axis(1)))
        })
        .with_n_features(2)
        .with_n_outputs(1);
        let cached = CachedModel::new(model, 4).unwrap();

        let first = cached
            .predict(array![[1., 2.], [3., 4.], [1., 2.]].view())
            .unwrap();
        assert_eq!(first, array![[3.], [7.], [3.]]);
        assert_eq!(calls.get(), 1);
        assert_eq!(rows.get(), 2);

        let second = cached.predict(array![[3., 4.], [5., 6.]].view()).unwrap();
        assert_eq!(second, array![[7.], [11.]]);
        assert_eq!(calls.get(), 2);
        assert_eq!(rows.get(), 3);
        assert_eq!(cached.len().unwrap(), 3);
    }

    #[test]
    fn cached_model_has_bounded_lru_eviction_and_can_be_cleared() {
        let rows = Cell::new(0usize);
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            rows.set(rows.get() + x.nrows());
            Ok(x.to_owned())
        });
        let cached = CachedModel::new(model, 2).unwrap();
        cached.predict(array![[1.], [2.], [3.]].view()).unwrap();
        assert_eq!(cached.len().unwrap(), 2);
        cached.predict(array![[1.]].view()).unwrap();
        assert_eq!(rows.get(), 4);
        cached.clear().unwrap();
        assert!(cached.is_empty().unwrap());
    }

    #[test]
    fn cached_model_rejects_zero_capacity() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| -> Result<Array2<f64>> { Ok(x.to_owned()) });
        assert!(CachedModel::new(model, 0).is_err());
    }
}
