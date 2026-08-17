use crate::{Background, DeepAttribution, Explainer, Explanation, Result, ShapError};
use ndarray::{Array2, ArrayView2, Axis, Slice};
/// Deep SHAP coordinator for framework adapters implementing multiplier
/// propagation through their native computation graph.
pub struct DeepExplainer<M> {
    model: M,
    background: Background,
    check_additivity: bool,
    tolerance: f64,
    batch_size: usize,
}
impl<M> DeepExplainer<M> {
    pub fn new(model: M, background: Background) -> Self {
        Self {
            model,
            background,
            check_additivity: true,
            tolerance: 1e-5,
            batch_size: 256,
        }
    }
    pub fn with_additivity_check(mut self, enabled: bool) -> Self {
        self.check_additivity = enabled;
        self
    }
    pub fn with_tolerance(mut self, t: f64) -> Self {
        self.tolerance = t;
        self
    }
    /// Limits explained samples submitted to the graph adapter in one call.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}
impl<M: DeepAttribution> Explainer for DeepExplainer<M> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        if !self.tolerance.is_finite() || self.tolerance < 0.0 || self.batch_size == 0 {
            return Err(ShapError::InvalidConfiguration(
                "Deep SHAP tolerance must be finite and non-negative and batch size positive"
                    .into(),
            ));
        }
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != self.background.n_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.background.n_features()),
                found: format!("{}", x.ncols()),
            });
        }
        if self
            .model
            .n_features()
            .is_some_and(|features| features != x.ncols())
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("model with {} features", x.ncols()),
                found: format!("model reports {:?}", self.model.n_features()),
            });
        }
        let bg_pred = self.model.predict(self.background.data())?;
        if bg_pred.nrows() != self.background.n_samples() || bg_pred.ncols() == 0 {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} background predictions", self.background.n_samples()),
                found: format!("{:?}", bg_pred.dim()),
            });
        }
        let base = bg_pred.mean_axis(Axis(0)).unwrap();
        crate::error::checked_f64_shape(&[x.nrows(), x.ncols(), base.len()], "deep explanation")?;
        if self
            .model
            .n_outputs()
            .is_some_and(|outputs| outputs != base.len())
        {
            return Err(ShapError::OutputDimensionMismatch {
                expected: self.model.n_outputs().unwrap(),
                found: base.len(),
            });
        }
        let bases = Array2::from_shape_fn((x.nrows(), base.len()), |(_, o)| base[o]);
        let mut parts = Vec::new();
        for start in (0..x.nrows()).step_by(self.batch_size) {
            let end = start.saturating_add(self.batch_size).min(x.nrows());
            parts.push(self.model.deep_contributions(
                x.slice_axis(Axis(0), Slice::from(start..end)),
                self.background.data(),
            )?);
        }
        let views = parts.iter().map(|part| part.view()).collect::<Vec<_>>();
        let values = ndarray::concatenate(Axis(0), &views)
            .map_err(|error| ShapError::ModelError(error.to_string()))?;
        let e = Explanation::new(values, bases, x.to_owned())?;
        if self.check_additivity {
            let mut predictions = Vec::new();
            for start in (0..x.nrows()).step_by(self.batch_size) {
                let end = start.saturating_add(self.batch_size).min(x.nrows());
                predictions.push(
                    self.model
                        .predict(x.slice_axis(Axis(0), Slice::from(start..end)))?,
                );
            }
            let views = predictions
                .iter()
                .map(|part| part.view())
                .collect::<Vec<_>>();
            let prediction = ndarray::concatenate(Axis(0), &views)
                .map_err(|error| ShapError::ModelError(error.to_string()))?;
            crate::metrics::check_additivity(&e, prediction.view(), self.tolerance)?
        }
        Ok(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DeepAttribution, Predict};
    use ndarray::{array, Array3};
    struct Adapter;
    impl Predict for Adapter {
        fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1)))
        }
    }
    impl DeepAttribution for Adapter {
        fn deep_contributions(
            &self,
            x: ArrayView2<'_, f64>,
            bg: ArrayView2<'_, f64>,
        ) -> Result<Array3<f64>> {
            let mean = bg.mean_axis(Axis(0)).unwrap();
            Ok(Array3::from_shape_fn(
                (x.nrows(), x.ncols(), 1),
                |(i, j, _)| x[[i, j]] - mean[j],
            ))
        }
    }
    #[test]
    fn validates_adapter_contributions() {
        let e = DeepExplainer::new(
            Adapter,
            Background::new(array![[0., 0.], [2., 2.]]).unwrap(),
        )
        .explain(array![[3., 4.]].view())
        .unwrap();
        assert_eq!(e.base_values()[[0, 0]], 2.);
        assert_eq!(e.reconstructed()[[0, 0]], 7.);
    }

    #[test]
    fn rejects_invalid_tolerance() {
        let result = DeepExplainer::new(Adapter, Background::new(array![[0., 0.]]).unwrap())
            .with_tolerance(f64::NAN)
            .explain(array![[1., 1.]].view());
        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }

    #[test]
    fn mini_batches_contributions_and_predictions() {
        let e = DeepExplainer::new(Adapter, Background::new(array![[0., 0.]]).unwrap())
            .with_batch_size(1)
            .explain(array![[1., 2.], [3., 4.], [5., 6.]].view())
            .unwrap();
        assert_eq!(e.values().dim(), (3, 2, 1));
        assert_eq!(e.reconstructed()[[2, 0]], 11.0);
    }
}
