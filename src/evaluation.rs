use crate::{coalition, Masker, Predict, Result, ShapError};
use ndarray::{Array2, ArrayView1, Axis, Slice};
use std::collections::{HashMap, HashSet, VecDeque};

/// Shared limits for model-agnostic explainer evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct EvaluationConfig {
    /// Maximum coalitions combined into one model call.
    pub coalition_batch_size: usize,
    /// Maximum cached coalition results for one explained sample.
    pub cache_capacity: usize,
    /// Optional hard limit on total model rows evaluated per sample.
    pub max_model_rows: Option<usize>,
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
        let mut batch = Array2::zeros((rows, self.masker.n_features()));
        let mut offset = 0;
        for part in &masked {
            let end = offset + part.nrows();
            batch
                .slice_axis_mut(Axis(0), Slice::from(offset..end))
                .assign(part);
            offset = end
        }
        let predictions = self.model.predict(batch.view())?;
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
        while self.cache.len() + masks.len() > self.config.cache_capacity {
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
    use crate::{Background, FnModel, IndependentMasker};
    use ndarray::{array, Array2, ArrayView2};
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
}
