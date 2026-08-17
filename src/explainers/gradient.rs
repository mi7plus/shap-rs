use crate::{
    Background, DifferentiablePredict, Explainer, Explanation, Result, ShapError,
    UncertainExplanation,
};
use ndarray::{Array2, Array3, ArrayView2, Axis};
use rand::{rngs::StdRng, Rng, SeedableRng};
/// Expected Gradients (Gradient SHAP) with background interpolation and
/// optional Gaussian local smoothing.
pub struct GradientExplainer<M> {
    model: M,
    background: Background,
    nsamples: usize,
    seed: u64,
    local_smoothing: f64,
}
impl<M> GradientExplainer<M> {
    pub fn new(model: M, background: Background) -> Self {
        Self {
            model,
            background,
            nsamples: 256,
            seed: 0,
            local_smoothing: 0.0,
        }
    }
    pub fn with_nsamples(mut self, n: usize) -> Self {
        self.nsamples = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    pub fn with_local_smoothing(mut self, s: f64) -> Self {
        self.local_smoothing = s;
        self
    }
}
impl<M: DifferentiablePredict> GradientExplainer<M> {
    /// Repeats Expected Gradients with independent deterministic seeds and
    /// returns the standard error of the mean attribution.
    pub fn explain_with_uncertainty(
        &self,
        x: ArrayView2<'_, f64>,
        repeats: usize,
    ) -> Result<UncertainExplanation> {
        if repeats < 2 {
            return Err(ShapError::InvalidConfiguration(
                "uncertainty estimation requires at least two repeats".into(),
            ));
        }
        let mut runs = Vec::with_capacity(repeats);
        for repeat in 0..repeats {
            runs.push(
                GradientExplainer {
                    model: &self.model,
                    background: self.background.clone(),
                    nsamples: self.nsamples,
                    seed: self.seed.wrapping_add(repeat as u64),
                    local_smoothing: self.local_smoothing,
                }
                .explain(x)?,
            );
        }
        let shape = runs[0].values().dim();
        let mut mean = Array3::<f64>::zeros(shape);
        for run in &runs {
            ndarray::Zip::from(&mut mean)
                .and(run.values())
                .for_each(|average, &value| *average += value);
        }
        mean.mapv_inplace(|value| value / repeats as f64);
        let mut variance = Array3::<f64>::zeros(shape);
        for run in &runs {
            ndarray::Zip::from(&mut variance)
                .and(run.values())
                .and(&mean)
                .for_each(|sum, &value, &average| *sum += (value - average).powi(2));
        }
        let standard_errors =
            variance.mapv(|value| (value / ((repeats - 1) * repeats) as f64).sqrt());
        let explanation = Explanation::new(mean, runs[0].base_values().to_owned(), x.to_owned())?;
        UncertainExplanation::new(explanation, standard_errors, repeats)
    }
}
impl<M: DifferentiablePredict> Explainer for GradientExplainer<M> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.background.n_features();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features"),
                found: format!("{}", x.ncols()),
            });
        }
        if self.nsamples == 0 || !self.local_smoothing.is_finite() || self.local_smoothing < 0. {
            return Err(ShapError::InvalidConfiguration(
                "nsamples must be positive and local smoothing non-negative".into(),
            ));
        }
        let prediction = self.model.predict(self.background.data())?;
        if prediction.nrows() != self.background.n_samples() || prediction.ncols() == 0 {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} background predictions", self.background.n_samples()),
                found: format!("{:?}", prediction.dim()),
            });
        }
        let base = prediction.mean_axis(Axis(0)).unwrap();
        let o = base.len();
        let bases = Array2::from_shape_fn((x.nrows(), o), |(_, k)| base[k]);
        let std = feature_std(&self.background);
        let mut values = Array3::zeros((x.nrows(), m, o));
        for n in 0..x.nrows() {
            let mut rng = StdRng::seed_from_u64(crate::coalition::sample_seed(self.seed, x.row(n)));
            let mut points = Array2::zeros((self.nsamples, m));
            let mut deltas = Array2::zeros((self.nsamples, m));
            for s in 0..self.nsamples {
                let b = rng.gen_range(0..self.background.n_samples());
                let alpha = rng.gen::<f64>();
                for j in 0..m {
                    let noise = if self.local_smoothing > 0. {
                        gaussian(&mut rng) * self.local_smoothing * std[j]
                    } else {
                        0.
                    };
                    let delta = x[[n, j]] + noise - self.background.data()[[b, j]];
                    deltas[[s, j]] = delta;
                    points[[s, j]] = self.background.data()[[b, j]] + alpha * delta
                }
            }
            let gradients = self.model.gradients(points.view())?;
            if gradients.dim() != (self.nsamples, m, o) {
                return Err(ShapError::DimensionMismatch {
                    expected: format!("({}, {m}, {o}) gradients", self.nsamples),
                    found: format!("{:?}", gradients.dim()),
                });
            }
            if gradients.iter().any(|v| !v.is_finite()) {
                return Err(ShapError::ModelError(
                    "gradient contains a non-finite value".into(),
                ));
            }
            for j in 0..m {
                for k in 0..o {
                    values[[n, j, k]] = (0..self.nsamples)
                        .map(|s| gradients[[s, j, k]] * deltas[[s, j]])
                        .sum::<f64>()
                        / self.nsamples as f64
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
    }
}
fn gaussian<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.gen::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.gen::<f64>();
    (-2. * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}
fn feature_std(bg: &Background) -> Vec<f64> {
    let mean = bg.data().mean_axis(Axis(0)).unwrap();
    (0..bg.n_features())
        .map(|j| {
            (bg.data()
                .column(j)
                .iter()
                .map(|x| (x - mean[j]).powi(2))
                .sum::<f64>()
                / bg.n_samples() as f64)
                .sqrt()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Predict;
    use ndarray::array;
    struct Linear;
    impl Predict for Linear {
        fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            Ok(x.map_axis(Axis(1), |r| 2. * r[0] - r[1])
                .insert_axis(Axis(1)))
        }
    }
    impl DifferentiablePredict for Linear {
        fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>> {
            Ok(Array3::from_shape_fn((x.nrows(), 2, 1), |(_, j, _)| {
                if j == 0 {
                    2.
                } else {
                    -1.
                }
            }))
        }
    }
    #[test]
    fn expected_gradients_is_exact_for_linear_models() {
        let e = GradientExplainer::new(Linear, Background::new(array![[0., 0.]]).unwrap())
            .with_nsamples(32)
            .explain(array![[3., 4.]].view())
            .unwrap();
        assert!((e.values()[[0, 0, 0]] - 6.).abs() < 1e-12);
        assert!((e.values()[[0, 1, 0]] + 4.).abs() < 1e-12);
        assert!((e.reconstructed()[[0, 0]] - 2.).abs() < 1e-12);
    }

    #[test]
    fn reports_uncertainty_for_stochastic_expected_gradients() {
        let e =
            GradientExplainer::new(Linear, Background::new(array![[0., 0.], [2., 4.]]).unwrap())
                .with_nsamples(16)
                .explain_with_uncertainty(array![[3., 4.]].view(), 4)
                .unwrap();
        assert_eq!(e.repeats(), 4);
        assert_eq!(e.standard_errors().dim(), (1, 2, 1));
        assert!(e.standard_errors().iter().all(|value| value.is_finite()));
    }
}
