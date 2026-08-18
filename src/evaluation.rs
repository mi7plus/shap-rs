use crate::{coalition, Masker, Predict, Result, ShapError};
use ndarray::{Array2, ArrayView1, Axis, Slice};
use std::collections::{HashMap, HashSet, VecDeque};

/// Shared limits for model-agnostic explainer evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(try_from = "EvaluationConfigPayload")]
pub struct EvaluationConfig {
    /// Maximum coalitions combined into one model call.
    pub coalition_batch_size: usize,
    /// Maximum cached coalition results for one explained sample.
    pub cache_capacity: usize,
    /// Optional hard limit on total model rows evaluated per sample.
    pub max_model_rows: Option<usize>,
}
#[derive(serde::Deserialize)]
struct EvaluationConfigPayload {
    coalition_batch_size: usize,
    cache_capacity: usize,
    max_model_rows: Option<usize>,
}
impl TryFrom<EvaluationConfigPayload> for EvaluationConfig {
    type Error = ShapError;
    fn try_from(payload: EvaluationConfigPayload) -> Result<Self> {
        Self {
            coalition_batch_size: payload.coalition_batch_size,
            cache_capacity: payload.cache_capacity,
            max_model_rows: payload.max_model_rows,
        }
        .validate()
    }
}
impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            coalition_batch_size: 64,
            cache_capacity: 4096,
            max_model_rows: None,
        }
    }
}
impl EvaluationConfig {
    pub fn validate(self) -> Result<Self> {
        if self.coalition_batch_size == 0 || self.cache_capacity == 0 {
            return Err(ShapError::InvalidConfiguration(
                "coalition batch size and cache capacity must be positive".into(),
            ));
        }
        if self.coalition_batch_size > self.cache_capacity {
            return Err(ShapError::InvalidConfiguration(
                "cache capacity must be at least the coalition batch size".into(),
            ));
        }
        if self.max_model_rows == Some(0) {
            return Err(ShapError::InvalidConfiguration(
                "model row evaluation limit must be positive when configured".into(),
            ));
        }
        Ok(self)
    }
}

pub(crate) struct CoalitionEvaluator<'a, M, K> {
    model: &'a M,
    masker: &'a K,
    config: EvaluationConfig,
    cache: HashMap<u64, Vec<f64>>,
    cache_order: VecDeque<u64>,
    rows_evaluated: usize,
    outputs: Option<usize>,
}
impl<'a, M: Predict, K: Masker> CoalitionEvaluator<'a, M, K> {
    pub(crate) fn new(model: &'a M, masker: &'a K, config: EvaluationConfig) -> Result<Self> {
        Ok(Self {
            model,
            masker,
            config: config.validate()?,
            cache: HashMap::new(),
            cache_order: VecDeque::new(),
            rows_evaluated: 0,
            outputs: None,
        })
    }
    pub(crate) fn evaluate(
        &mut self,
        sample: ArrayView1<'_, f64>,
        masks: &[u64],
    ) -> Result<Vec<Vec<f64>>> {
        let mut results = HashMap::new();
        let mut missing = Vec::new();
        let mut missing_set = HashSet::new();
        for &mask in masks {
            if let Some(value) = self.cache.get(&mask).cloned() {
                self.touch(mask);
                results.insert(mask, value);
            } else if missing_set.insert(mask) {
                missing.push(mask);
            }
        }
        for chunk in missing.chunks(self.config.coalition_batch_size) {
            self.evaluate_chunk(sample, chunk)?;
            for &mask in chunk {
                let value =
                    self.cache.get(&mask).cloned().ok_or_else(|| {
                        ShapError::Other("coalition cache invariant failed".into())
                    })?;
                results.insert(mask, value);
            }
        }
        masks
            .iter()
            .map(|m| {
                results
                    .get(m)
                    .cloned()
                    .ok_or_else(|| ShapError::Other("coalition cache invariant failed".into()))
            })
            .collect()
    }
    fn evaluate_chunk(&mut self, sample: ArrayView1<'_, f64>, masks: &[u64]) -> Result<()> {
        if masks.is_empty() {
            return Ok(());
        }
        if self.masker.streams_masked_batches() {
            for &mask in masks {
                self.evaluate_streaming_mask(sample, mask)?;
            }
            return Ok(());
        }
        let mut masked = Vec::with_capacity(masks.len());
        let mut rows = 0usize;
        for &mask in masks {
            let part = self
                .masker
                .mask(sample, &coalition::members(mask, self.masker.n_features()))?;
            if part.nrows() == 0 {
                return Err(ShapError::MaskerError("masker returned no rows".into()));
            }
            rows = rows.checked_add(part.nrows()).ok_or_else(|| {
                ShapError::InvalidConfiguration("coalition batch is too large".into())
            })?;
            masked.push(part)
        }
        if self
            .config
            .max_model_rows
            .is_some_and(|limit| self.rows_evaluated.saturating_add(rows) > limit)
        {
            return Err(ShapError::InvalidConfiguration(
                "model row evaluation limit exceeded".into(),
            ));
        }
        crate::error::checked_f64_shape(
            &[rows, self.masker.n_input_features()],
            "masked coalition batch",
        )?;
        let mut batch = Array2::zeros((rows, self.masker.n_input_features()));
        let mut offset = 0;
        for part in &masked {
            let end = offset + part.nrows();
            batch
                .slice_axis_mut(Axis(0), Slice::from(offset..end))
                .assign(part);
            offset = end
        }
        let predictions = self.model.predict_owned(batch)?;
        if predictions.nrows() != rows || predictions.ncols() == 0 {
            return Err(ShapError::DimensionMismatch {
                expected: format!("({rows}, outputs>0)"),
                found: format!("{:?}", predictions.dim()),
            });
        }
        if self.outputs.is_some_and(|o| o != predictions.ncols()) {
            return Err(ShapError::OutputDimensionMismatch {
                expected: self.outputs.unwrap(),
                found: predictions.ncols(),
            });
        }
        if predictions.iter().any(|v| !v.is_finite()) {
            return Err(ShapError::ModelError(
                "prediction contains a non-finite value".into(),
            ));
        }
        self.outputs = Some(predictions.ncols());
        self.rows_evaluated += rows;
        while masks.len() > self.config.cache_capacity.saturating_sub(self.cache.len()) {
            let Some(key) = self.cache_order.pop_front() else {
                break;
            };
            self.cache.remove(&key);
        }
        let mut offset = 0;
        for (i, &mask) in masks.iter().enumerate() {
            let end = offset + masked[i].nrows();
            let value = predictions
                .slice_axis(Axis(0), Slice::from(offset..end))
                .mean_axis(Axis(0))
                .unwrap()
                .to_vec();
            offset = end;
            self.cache.insert(mask, value);
            self.cache_order.push_back(mask);
        }
        Ok(())
    }

    fn evaluate_streaming_mask(&mut self, sample: ArrayView1<'_, f64>, mask: u64) -> Result<()> {
        let members = coalition::members(mask, self.masker.n_features());
        let model = self.model;
        let starting_rows = self.rows_evaluated;
        let row_limit = self.config.max_model_rows;
        let mut rows = 0usize;
        let mut outputs = self.outputs;
        let mut sums = Vec::<f64>::new();
        self.masker
            .for_each_masked_batch(sample, &members, &mut |batch| {
                let batch_rows = batch.nrows();
                let next_rows = rows.checked_add(batch_rows).ok_or_else(|| {
                    ShapError::InvalidConfiguration("streaming model row count overflow".into())
                })?;
                if row_limit.is_some_and(|limit| starting_rows.saturating_add(next_rows) > limit) {
                    return Err(ShapError::InvalidConfiguration(
                        "model row evaluation limit exceeded".into(),
                    ));
                }
                let predictions = model.predict_owned(batch)?;
                if predictions.nrows() != batch_rows || predictions.ncols() == 0 {
                    return Err(ShapError::DimensionMismatch {
                        expected: format!("({batch_rows}, outputs>0)"),
                        found: format!("{:?}", predictions.dim()),
                    });
                }
                if let Some(expected) = outputs {
                    if expected != predictions.ncols() {
                        return Err(ShapError::OutputDimensionMismatch {
                            expected,
                            found: predictions.ncols(),
                        });
                    }
                } else {
                    outputs = Some(predictions.ncols());
                    sums.resize(predictions.ncols(), 0.0);
                }
                if predictions.iter().any(|value| !value.is_finite()) {
                    return Err(ShapError::ModelError(
                        "prediction contains a non-finite value".into(),
                    ));
                }
                if sums.is_empty() {
                    sums.resize(predictions.ncols(), 0.0);
                }
                for prediction in predictions.rows() {
                    for (sum, value) in sums.iter_mut().zip(prediction) {
                        *sum += *value;
                    }
                }
                rows = next_rows;
                Ok(())
            })?;
        if rows == 0 {
            return Err(ShapError::MaskerError(
                "streaming masker returned no rows".into(),
            ));
        }
        let value = sums
            .into_iter()
            .map(|sum| sum / rows as f64)
            .collect::<Vec<_>>();
        if value.iter().any(|value| !value.is_finite()) {
            return Err(ShapError::ModelError(
                "streaming prediction mean is non-finite".into(),
            ));
        }
        self.rows_evaluated = starting_rows.checked_add(rows).ok_or_else(|| {
            ShapError::InvalidConfiguration("model row evaluation count overflow".into())
        })?;
        self.outputs = outputs;
        while self.cache.len() >= self.config.cache_capacity {
            let Some(key) = self.cache_order.pop_front() else {
                break;
            };
            self.cache.remove(&key);
        }
        self.cache.insert(mask, value);
        self.cache_order.push_back(mask);
        Ok(())
    }
    fn touch(&mut self, mask: u64) {
        if let Some(position) = self.cache_order.iter().position(|&key| key == mask) {
            self.cache_order.remove(position);
        }
        self.cache_order.push_back(mask);
    }
    #[cfg(test)]
    pub(crate) fn rows_evaluated(&self) -> usize {
        self.rows_evaluated
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Background, FnModel, FnStreamingMasker, IndependentMasker};
    use ndarray::{array, Array2, ArrayView1, ArrayView2};
    use std::cell::Cell;
    #[test]
    fn batches_and_deduplicates_coalitions() {
        let calls = Cell::new(0);
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            calls.set(calls.get() + 1);
            Ok(Array2::from_shape_fn((x.nrows(), 1), |(i, _)| {
                x.row(i).sum()
            }))
        });
        let masker = IndependentMasker::new(Background::new(array![[0., 0.], [1., 1.]]).unwrap());
        let config = EvaluationConfig {
            coalition_batch_size: 8,
            cache_capacity: 8,
            max_model_rows: None,
        };
        let mut evaluator = CoalitionEvaluator::new(&model, &masker, config).unwrap();
        let values = evaluator
            .evaluate(array![2., 3.].view(), &[0, 1, 2, 3, 1])
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(values[1], values[4]);
        assert_eq!(evaluator.rows_evaluated(), 8);
    }
    #[test]
    fn coalition_batch_uses_owned_prediction_fast_path_once() {
        struct OwnedTrackingModel {
            borrowed: Cell<usize>,
            owned: Cell<usize>,
        }
        impl Predict for OwnedTrackingModel {
            fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
                self.borrowed.set(self.borrowed.get() + 1);
                Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1)))
            }
            fn predict_owned(&self, x: Array2<f64>) -> Result<Array2<f64>> {
                self.owned.set(self.owned.get() + 1);
                Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1)))
            }
        }
        let model = OwnedTrackingModel {
            borrowed: Cell::new(0),
            owned: Cell::new(0),
        };
        let masker = IndependentMasker::new(Background::new(array![[0., 0.], [1., 1.]]).unwrap());
        let mut evaluator = CoalitionEvaluator::new(
            &model,
            &masker,
            EvaluationConfig {
                coalition_batch_size: 4,
                cache_capacity: 4,
                max_model_rows: None,
            },
        )
        .unwrap();
        evaluator
            .evaluate(array![2., 3.].view(), &[0, 1, 2, 3])
            .unwrap();
        assert_eq!(model.owned.get(), 1);
        assert_eq!(model.borrowed.get(), 0);
    }
    #[test]
    fn bounded_cache_does_not_limit_request_size() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(Array2::from_shape_fn((x.nrows(), 1), |(i, _)| {
                x.row(i).sum()
            }))
        });
        let masker = IndependentMasker::new(Background::new(array![[0., 0.]]).unwrap());
        let mut evaluator = CoalitionEvaluator::new(
            &model,
            &masker,
            EvaluationConfig {
                coalition_batch_size: 2,
                cache_capacity: 2,
                max_model_rows: None,
            },
        )
        .unwrap();
        let values = evaluator
            .evaluate(array![2., 3.].view(), &[0, 1, 2, 3])
            .unwrap();
        assert_eq!(values.len(), 4);
        assert_eq!(evaluator.cache.len(), 2);
    }
    #[test]
    fn cache_uses_deterministic_lru_eviction() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let masker = IndependentMasker::new(Background::new(array![[0., 0.]]).unwrap());
        let mut evaluator = CoalitionEvaluator::new(
            &model,
            &masker,
            EvaluationConfig {
                coalition_batch_size: 1,
                cache_capacity: 2,
                max_model_rows: None,
            },
        )
        .unwrap();
        evaluator.evaluate(array![2., 3.].view(), &[0, 1]).unwrap();
        evaluator.evaluate(array![2., 3.].view(), &[0]).unwrap();
        evaluator.evaluate(array![2., 3.].view(), &[2]).unwrap();
        assert!(evaluator.cache.contains_key(&0));
        assert!(evaluator.cache.contains_key(&2));
        assert!(!evaluator.cache.contains_key(&1));
    }

    #[test]
    fn rejects_zero_model_row_budget() {
        assert!(EvaluationConfig {
            coalition_batch_size: 1,
            cache_capacity: 1,
            max_model_rows: Some(0),
        }
        .validate()
        .is_err());
    }

    #[test]
    fn consumes_streaming_masker_batches_without_collecting_background() {
        let calls = Cell::new(0usize);
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            calls.set(calls.get() + 1);
            Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1)))
        });
        let masker = FnStreamingMasker::new(
            2,
            |sample: ArrayView1<'_, f64>,
             present: &[bool],
             visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>| {
                for mut batch in [array![[0., 0.], [2., 2.]], array![[4., 4.]]] {
                    for (feature, enabled) in present.iter().copied().enumerate() {
                        if enabled {
                            batch.column_mut(feature).fill(sample[feature]);
                        }
                    }
                    visitor(batch)?;
                }
                Ok(())
            },
        )
        .unwrap();
        let mut evaluator = CoalitionEvaluator::new(
            &model,
            &masker,
            EvaluationConfig {
                coalition_batch_size: 4,
                cache_capacity: 4,
                max_model_rows: None,
            },
        )
        .unwrap();
        let values = evaluator
            .evaluate(array![10., 20.].view(), &[0, 3])
            .unwrap();
        assert_eq!(values, vec![vec![4.], vec![30.]]);
        assert_eq!(calls.get(), 4);
        assert_eq!(evaluator.rows_evaluated(), 6);
    }

    #[test]
    fn streaming_batches_respect_the_total_model_row_budget() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.to_owned()));
        let masker = FnStreamingMasker::new(
            1,
            |_: ArrayView1<'_, f64>,
             _: &[bool],
             visitor: &mut dyn FnMut(Array2<f64>) -> Result<()>| {
                visitor(array![[0.]])?;
                visitor(array![[1.]])
            },
        )
        .unwrap();
        let mut evaluator = CoalitionEvaluator::new(
            &model,
            &masker,
            EvaluationConfig {
                coalition_batch_size: 1,
                cache_capacity: 1,
                max_model_rows: Some(1),
            },
        )
        .unwrap();
        assert!(evaluator.evaluate(array![2.].view(), &[0]).is_err());
    }
}
