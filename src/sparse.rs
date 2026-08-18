//! Native CSR model boundary and sparse permutation SHAP.

use crate::{EvaluationConfig, Explanation, Link, Result, ShapError};
use ndarray::{Array2, Array3, ArrayView2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Validated compressed-sparse-row matrix.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SparseMatrix {
    rows: usize,
    columns: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct SparseMatrixPayload {
    rows: usize,
    columns: usize,
    indptr: Vec<usize>,
    indices: Vec<usize>,
    values: Vec<f64>,
}

impl<'de> Deserialize<'de> for SparseMatrix {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let payload = SparseMatrixPayload::deserialize(deserializer)?;
        Self::new(
            payload.rows,
            payload.columns,
            payload.indptr,
            payload.indices,
            payload.values,
        )
        .map_err(serde::de::Error::custom)
    }
}

impl SparseMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        indptr: Vec<usize>,
        indices: Vec<usize>,
        values: Vec<f64>,
    ) -> Result<Self> {
        let matrix = Self {
            rows,
            columns,
            indptr,
            indices,
            values,
        };
        matrix.validate()?;
        Ok(matrix)
    }

    pub fn from_dense(dense: ArrayView2<'_, f64>) -> Result<Self> {
        if dense.ncols() == 0 {
            return Err(ShapError::InvalidConfiguration(
                "sparse matrices must contain at least one column".into(),
            ));
        }
        let mut indptr = Vec::with_capacity(dense.nrows().saturating_add(1));
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.push(0);
        for row in dense.rows() {
            for (column, value) in row.iter().copied().enumerate() {
                if value != 0.0 {
                    indices.push(column);
                    values.push(value);
                }
            }
            indptr.push(indices.len());
        }
        Self::new(dense.nrows(), dense.ncols(), indptr, indices, values)
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }

    pub fn ncols(&self) -> usize {
        self.columns
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn indptr(&self) -> &[usize] {
        &self.indptr
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    pub fn values(&self) -> &[f64] {
        &self.values
    }

    pub fn validate(&self) -> Result<()> {
        if self.columns == 0 {
            return Err(ShapError::InvalidConfiguration(
                "sparse matrices must contain at least one column".into(),
            ));
        }
        if self.indptr.len() != self.rows.saturating_add(1)
            || self.indptr.first().copied() != Some(0)
            || self.indptr.last().copied() != Some(self.indices.len())
            || self.indices.len() != self.values.len()
        {
            return Err(ShapError::InvalidConfiguration(
                "invalid CSR pointer or value lengths".into(),
            ));
        }
        if self.values.contains(&0.0) {
            return Err(ShapError::InvalidConfiguration(
                "canonical CSR values must not contain explicit zeros".into(),
            ));
        }
        for row in 0..self.rows {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            if start > end || end > self.indices.len() {
                return Err(ShapError::InvalidConfiguration(
                    "CSR row pointers must be monotonic and in bounds".into(),
                ));
            }
            let row_indices = &self.indices[start..end];
            if row_indices.iter().any(|index| *index >= self.columns)
                || row_indices.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return Err(ShapError::InvalidConfiguration(
                    "CSR column indices must be sorted, unique, and in bounds".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn row(&self, row: usize) -> Result<SparseRowView<'_>> {
        if row >= self.rows {
            return Err(ShapError::InvalidSampleIndex {
                index: row,
                n_samples: self.rows,
            });
        }
        let range = self.indptr[row]..self.indptr[row + 1];
        Ok(SparseRowView {
            columns: self.columns,
            indices: &self.indices[range.clone()],
            values: &self.values[range],
        })
    }

    pub fn to_dense(&self) -> Result<Array2<f64>> {
        crate::error::checked_f64_shape(&[self.rows, self.columns], "dense sparse-matrix view")?;
        let mut dense = Array2::zeros((self.rows, self.columns));
        for row in 0..self.rows {
            let sparse = self.row(row)?;
            for (&column, &value) in sparse.indices.iter().zip(sparse.values) {
                dense[[row, column]] = value;
            }
        }
        Ok(dense)
    }

    fn concatenate_rows(parts: &[Self]) -> Result<Self> {
        let columns = parts.first().map_or(0, Self::ncols);
        if columns == 0 || parts.iter().any(|part| part.ncols() != columns) {
            return Err(ShapError::DimensionMismatch {
                expected: format!("sparse batches with {columns} columns"),
                found: "incompatible sparse batch columns".into(),
            });
        }
        let rows = parts.iter().try_fold(0usize, |total, part| {
            total.checked_add(part.nrows()).ok_or_else(|| {
                ShapError::InvalidConfiguration("sparse batch row count overflow".into())
            })
        })?;
        let nnz = parts.iter().try_fold(0usize, |total, part| {
            total.checked_add(part.nnz()).ok_or_else(|| {
                ShapError::InvalidConfiguration("sparse batch nonzero count overflow".into())
            })
        })?;
        let pointers = rows.checked_add(1).ok_or_else(|| {
            ShapError::InvalidConfiguration("sparse batch pointer count overflow".into())
        })?;
        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr.try_reserve_exact(pointers).map_err(|error| {
            ShapError::InvalidConfiguration(format!(
                "cannot allocate sparse batch row pointers: {error}"
            ))
        })?;
        indices.try_reserve_exact(nnz).map_err(|error| {
            ShapError::InvalidConfiguration(format!(
                "cannot allocate sparse batch indices: {error}"
            ))
        })?;
        values.try_reserve_exact(nnz).map_err(|error| {
            ShapError::InvalidConfiguration(format!("cannot allocate sparse batch values: {error}"))
        })?;
        indptr.push(0);
        for part in parts {
            for row in 0..part.nrows() {
                let range = part.indptr[row]..part.indptr[row + 1];
                indices.extend_from_slice(&part.indices[range.clone()]);
                values.extend_from_slice(&part.values[range]);
                indptr.push(indices.len());
            }
        }
        Self::new(rows, columns, indptr, indices, values)
    }
}

/// Borrowed CSR row.
#[derive(Debug, Clone, Copy)]
pub struct SparseRowView<'a> {
    columns: usize,
    indices: &'a [usize],
    values: &'a [f64],
}

impl SparseRowView<'_> {
    pub fn len(&self) -> usize {
        self.columns
    }

    pub fn is_empty(&self) -> bool {
        self.columns == 0
    }

    pub fn indices(&self) -> &[usize] {
        self.indices
    }

    pub fn values(&self) -> &[f64] {
        self.values
    }

    pub fn get(&self, column: usize) -> Result<f64> {
        if column >= self.columns {
            return Err(ShapError::InvalidFeatureIndex {
                index: column,
                n_features: self.columns,
            });
        }
        Ok(self
            .indices
            .binary_search(&column)
            .map(|position| self.values[position])
            .unwrap_or(0.0))
    }
}

/// Prediction contract for models with a native CSR input boundary.
pub trait SparsePredict {
    fn predict_sparse(&self, input: &SparseMatrix) -> Result<Array2<f64>>;
    fn n_features(&self) -> Option<usize> {
        None
    }
    fn n_outputs(&self) -> Option<usize> {
        None
    }
}

impl<T: SparsePredict + ?Sized> SparsePredict for &T {
    fn predict_sparse(&self, input: &SparseMatrix) -> Result<Array2<f64>> {
        (**self).predict_sparse(input)
    }
    fn n_features(&self) -> Option<usize> {
        (**self).n_features()
    }
    fn n_outputs(&self) -> Option<usize> {
        (**self).n_outputs()
    }
}

pub struct FnSparseModel<F> {
    predict_fn: F,
    n_features: Option<usize>,
    n_outputs: Option<usize>,
}

impl<F> FnSparseModel<F> {
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

impl<F> SparsePredict for FnSparseModel<F>
where
    F: Fn(&SparseMatrix) -> Result<Array2<f64>>,
{
    fn predict_sparse(&self, input: &SparseMatrix) -> Result<Array2<f64>> {
        (self.predict_fn)(input)
    }
    fn n_features(&self) -> Option<usize> {
        self.n_features
    }
    fn n_outputs(&self) -> Option<usize> {
        self.n_outputs
    }
}

/// Interventional masker backed by CSR background rows.
#[derive(Debug, Clone)]
pub struct SparseIndependentMasker {
    background: SparseMatrix,
}

impl SparseIndependentMasker {
    pub fn new(background: SparseMatrix) -> Result<Self> {
        if background.nrows() == 0 {
            return Err(ShapError::EmptyBackground);
        }
        Ok(Self { background })
    }

    pub fn background(&self) -> &SparseMatrix {
        &self.background
    }

    pub fn mask(&self, sample: SparseRowView<'_>, present: &[bool]) -> Result<SparseMatrix> {
        if sample.len() != self.background.ncols() || present.len() != self.background.ncols() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} sparse features", self.background.ncols()),
                found: format!("sample {}, mask {}", sample.len(), present.len()),
            });
        }
        let pointer_capacity = self.background.nrows().checked_add(1).ok_or_else(|| {
            ShapError::InvalidConfiguration("sparse mask row pointer count overflow".into())
        })?;
        let additional = self
            .background
            .nrows()
            .checked_mul(sample.indices.len())
            .and_then(|count| count.checked_add(self.background.nnz()))
            .ok_or_else(|| {
                ShapError::InvalidConfiguration("sparse masked nonzero bound overflow".into())
            })?;
        let mut indptr = Vec::new();
        let mut indices = Vec::new();
        let mut values = Vec::new();
        indptr
            .try_reserve_exact(pointer_capacity)
            .map_err(|error| {
                ShapError::InvalidConfiguration(format!(
                    "cannot allocate sparse mask row pointers: {error}"
                ))
            })?;
        indices.try_reserve(additional).map_err(|error| {
            ShapError::InvalidConfiguration(format!(
                "cannot allocate sparse masked indices: {error}"
            ))
        })?;
        values.try_reserve(additional).map_err(|error| {
            ShapError::InvalidConfiguration(format!(
                "cannot allocate sparse masked values: {error}"
            ))
        })?;
        indptr.push(0);
        for row in 0..self.background.nrows() {
            let background = self.background.row(row)?;
            let mut sample_position = 0;
            let mut background_position = 0;
            while sample_position < sample.indices.len()
                || background_position < background.indices.len()
            {
                let sample_column = sample.indices.get(sample_position).copied();
                let background_column = background.indices.get(background_position).copied();
                let column = match (sample_column, background_column) {
                    (Some(left), Some(right)) => left.min(right),
                    (Some(left), None) => left,
                    (None, Some(right)) => right,
                    (None, None) => break,
                };
                let sample_value = if sample_column == Some(column) {
                    let value = sample.values[sample_position];
                    sample_position += 1;
                    value
                } else {
                    0.0
                };
                let background_value = if background_column == Some(column) {
                    let value = background.values[background_position];
                    background_position += 1;
                    value
                } else {
                    0.0
                };
                let value = if present[column] {
                    sample_value
                } else {
                    background_value
                };
                if value != 0.0 {
                    indices.push(column);
                    values.push(value);
                }
            }
            indptr.push(indices.len());
        }
        SparseMatrix::new(
            self.background.nrows(),
            self.background.ncols(),
            indptr,
            indices,
            values,
        )
    }
}

struct SparseCoalitionEvaluator<'a, M> {
    model: &'a M,
    masker: &'a SparseIndependentMasker,
    config: EvaluationConfig,
    cache: HashMap<u64, Vec<f64>>,
    order: VecDeque<u64>,
    rows_evaluated: usize,
    outputs: Option<usize>,
}

impl<'a, M: SparsePredict> SparseCoalitionEvaluator<'a, M> {
    fn new(
        model: &'a M,
        masker: &'a SparseIndependentMasker,
        config: EvaluationConfig,
    ) -> Result<Self> {
        Ok(Self {
            model,
            masker,
            config: config.validate()?,
            cache: HashMap::new(),
            order: VecDeque::new(),
            rows_evaluated: 0,
            outputs: None,
        })
    }

    fn evaluate(&mut self, sample: SparseRowView<'_>, masks: &[u64]) -> Result<Vec<Vec<f64>>> {
        let mut result = HashMap::new();
        let mut missing = Vec::new();
        let mut seen = HashSet::new();
        for &mask in masks {
            if let Some(value) = self.cache.get(&mask).cloned() {
                self.touch(mask);
                result.insert(mask, value);
            } else if seen.insert(mask) {
                missing.push(mask);
            }
        }
        for chunk in missing.chunks(self.config.coalition_batch_size) {
            let parts = chunk
                .iter()
                .map(|mask| {
                    self.masker.mask(
                        sample,
                        &crate::coalition::members(*mask, self.masker.background.ncols()),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let batch = SparseMatrix::concatenate_rows(&parts)?;
            if self
                .config
                .max_model_rows
                .is_some_and(|limit| self.rows_evaluated.saturating_add(batch.nrows()) > limit)
            {
                return Err(ShapError::InvalidConfiguration(
                    "model row evaluation limit exceeded".into(),
                ));
            }
            if let Some(features) = self.model.n_features() {
                if features != batch.ncols() {
                    return Err(ShapError::DimensionMismatch {
                        expected: format!("{features} sparse model features"),
                        found: format!("{}", batch.ncols()),
                    });
                }
            }
            let predictions = self.model.predict_sparse(&batch)?;
            if predictions.nrows() != batch.nrows() || predictions.ncols() == 0 {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("({}, outputs>0)", batch.nrows()),
                    found: format!("{:?}", predictions.dim()),
                });
            }
            if predictions.iter().any(|value| !value.is_finite()) {
                return Err(ShapError::ModelError(
                    "sparse prediction contains a non-finite value".into(),
                ));
            }
            if let Some(outputs) = self.outputs {
                if outputs != predictions.ncols() {
                    return Err(ShapError::OutputDimensionMismatch {
                        expected: outputs,
                        found: predictions.ncols(),
                    });
                }
            } else {
                self.outputs = Some(predictions.ncols());
            }
            self.rows_evaluated =
                self.rows_evaluated
                    .checked_add(batch.nrows())
                    .ok_or_else(|| {
                        ShapError::InvalidConfiguration("sparse row count overflow".into())
                    })?;
            let mut offset = 0;
            for (&mask, part) in chunk.iter().zip(&parts) {
                let end = offset + part.nrows();
                let value = predictions
                    .slice_axis(Axis(0), ndarray::Slice::from(offset..end))
                    .mean_axis(Axis(0))
                    .unwrap()
                    .to_vec();
                offset = end;
                while self.cache.len() >= self.config.cache_capacity {
                    let Some(evicted) = self.order.pop_front() else {
                        break;
                    };
                    self.cache.remove(&evicted);
                }
                self.cache.insert(mask, value.clone());
                self.order.push_back(mask);
                result.insert(mask, value);
            }
        }
        masks
            .iter()
            .map(|mask| {
                result.get(mask).cloned().ok_or_else(|| {
                    ShapError::Other("sparse coalition cache invariant failed".into())
                })
            })
            .collect()
    }

    fn touch(&mut self, mask: u64) {
        if let Some(position) = self.order.iter().position(|value| *value == mask) {
            self.order.remove(position);
        }
        self.order.push_back(mask);
    }
}

/// Monte-Carlo permutation SHAP that keeps inputs, backgrounds, and coalition
/// batches in CSR form. Explanation display data is densified once at the end.
pub struct SparsePermutationExplainer<M> {
    model: M,
    masker: SparseIndependentMasker,
    n_permutations: usize,
    seed: u64,
    antithetic: bool,
    link: Link,
    evaluation: EvaluationConfig,
}

impl<M> SparsePermutationExplainer<M> {
    pub fn new(model: M, background: SparseMatrix) -> Result<Self> {
        Ok(Self {
            model,
            masker: SparseIndependentMasker::new(background)?,
            n_permutations: 128,
            seed: 0,
            antithetic: true,
            link: Link::Identity,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 65536,
                max_model_rows: None,
            },
        })
    }
    pub fn with_n_permutations(mut self, count: usize) -> Self {
        self.n_permutations = count;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_antithetic(mut self, enabled: bool) -> Self {
        self.antithetic = enabled;
        self
    }
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }
    pub fn with_evaluation_config(mut self, config: EvaluationConfig) -> Self {
        self.evaluation = config;
        self
    }
}

impl<M: SparsePredict> SparsePermutationExplainer<M> {
    pub fn explain(&self, input: &SparseMatrix) -> Result<Explanation> {
        input.validate()?;
        let features = self.masker.background.ncols();
        if input.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if input.ncols() != features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{features} sparse features"),
                found: format!("{}", input.ncols()),
            });
        }
        if features >= 63 {
            return Err(ShapError::InvalidConfiguration(
                "sparse permutation SHAP currently supports at most 62 features".into(),
            ));
        }
        if self.n_permutations == 0 {
            return Err(ShapError::InvalidConfiguration(
                "n_permutations must be positive".into(),
            ));
        }
        self.n_permutations.checked_mul(features).ok_or_else(|| {
            ShapError::InvalidConfiguration("sparse permutation step count overflow".into())
        })?;
        let mut probe = SparseCoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let outputs = probe.evaluate(input.row(0)?, &[0])?[0].len();
        crate::error::checked_f64_shape(
            &[input.nrows(), features, outputs],
            "sparse permutation explanation",
        )?;
        let mut values = Array3::zeros((input.nrows(), features, outputs));
        let mut bases = Array2::zeros((input.nrows(), outputs));
        for sample_index in 0..input.nrows() {
            let sample = input.row(sample_index)?;
            let mut rng = StdRng::seed_from_u64(sparse_sample_seed(self.seed, sample));
            let mut requested = vec![0u64];
            let mut steps = Vec::with_capacity(self.n_permutations * features);
            let mut generated = 0;
            while generated < self.n_permutations {
                let mut order = (0..features).collect::<Vec<_>>();
                order.shuffle(&mut rng);
                append_order(&order, &mut requested, &mut steps);
                generated += 1;
                if self.antithetic && generated < self.n_permutations {
                    order.reverse();
                    append_order(&order, &mut requested, &mut steps);
                    generated += 1;
                }
            }
            let mut evaluator =
                SparseCoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let evaluated = evaluator
                .evaluate(sample, &requested)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| self.link.forward(value))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            for output in 0..outputs {
                bases[[sample_index, output]] = evaluated[0][output];
            }
            for (feature, before, after) in steps {
                for output in 0..outputs {
                    values[[sample_index, feature, output]] += (evaluated[after][output]
                        - evaluated[before][output])
                        / self.n_permutations as f64;
                }
            }
        }
        Explanation::new(values, bases, input.to_dense()?)
    }
}

fn append_order(order: &[usize], requested: &mut Vec<u64>, steps: &mut Vec<(usize, usize, usize)>) {
    let mut mask = 0u64;
    let mut before = 0usize;
    for &feature in order {
        mask |= 1u64 << feature;
        requested.push(mask);
        let after = requested.len() - 1;
        steps.push((feature, before, after));
        before = after;
    }
}

fn sparse_sample_seed(seed: u64, sample: SparseRowView<'_>) -> u64 {
    fn mix(mut value: u64) -> u64 {
        value ^= value >> 30;
        value = value.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        value ^= value >> 27;
        value = value.wrapping_mul(0x94D0_49BB_1331_11EB);
        value ^ (value >> 31)
    }
    sample.indices.iter().zip(sample.values).fold(
        mix(seed ^ sample.columns as u64),
        |state, (&index, &value)| mix(state ^ mix(index as u64) ^ mix(value.to_bits())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::cell::Cell;

    #[test]
    fn csr_validation_and_dense_round_trip() {
        let dense = array![[0., 2., 0.], [3., 0., 4.]];
        let sparse = SparseMatrix::from_dense(dense.view()).unwrap();
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.to_dense().unwrap(), dense);
        assert!(SparseMatrix::new(1, 2, vec![0, 2], vec![1, 1], vec![2., 3.]).is_err());
    }

    #[test]
    fn sparse_masker_merges_rows_without_dense_coalitions() {
        let background =
            SparseMatrix::from_dense(array![[0., 2., 0.], [3., 0., 4.]].view()).unwrap();
        let sample_matrix = SparseMatrix::from_dense(array![[5., 0., 6.]].view()).unwrap();
        let masker = SparseIndependentMasker::new(background).unwrap();
        let masked = masker
            .mask(sample_matrix.row(0).unwrap(), &[true, false, true])
            .unwrap();
        assert_eq!(
            masked.to_dense().unwrap(),
            array![[5., 2., 6.], [5., 0., 6.]]
        );
        assert_eq!(masked.nnz(), 5);
    }

    #[test]
    fn sparse_permutation_matches_additive_model_without_dense_model_input() {
        let sparse_calls = Cell::new(0usize);
        let model = FnSparseModel::new(|input: &SparseMatrix| {
            sparse_calls.set(sparse_calls.get() + 1);
            Ok(Array2::from_shape_fn((input.nrows(), 1), |(row, _)| {
                let sparse = input.row(row).unwrap();
                sparse
                    .indices()
                    .iter()
                    .zip(sparse.values())
                    .map(|(&column, &value)| (column as f64 + 1.0) * value)
                    .sum()
            }))
        });
        let background = SparseMatrix::from_dense(array![[0., 0., 0.]].view()).unwrap();
        let input = SparseMatrix::from_dense(array![[2., 0., 4.]].view()).unwrap();
        let explanation = SparsePermutationExplainer::new(model, background)
            .unwrap()
            .with_n_permutations(2)
            .explain(&input)
            .unwrap();
        assert_eq!(explanation.values(), array![[[2.], [0.], [12.]]].view());
        assert_eq!(explanation.reconstructed(), array![[14.]]);
        assert!(sparse_calls.get() > 0);
    }
}
