use crate::{Background, Explainer, Explanation, Result, ShapError};
use ndarray::{Array1, Array2, Array3, ArrayView2};
/// Exact interventional SHAP values for `y = x W + intercept`.
pub struct LinearExplainer {
    coefficients: Array2<f64>,
    intercept: Array1<f64>,
    background: Background,
}

/// Exact linear SHAP under a multivariate-Gaussian feature distribution.
/// Missing features use their conditional expectation given present features.
pub struct CorrelatedLinearExplainer {
    coefficients: Array2<f64>,
    intercept: Array1<f64>,
    mean: Array1<f64>,
    covariance: Array2<f64>,
    max_features: usize,
    ridge: f64,
}
impl CorrelatedLinearExplainer {
    pub fn new(
        coefficients: Array2<f64>,
        intercept: Array1<f64>,
        mean: Array1<f64>,
        covariance: Array2<f64>,
    ) -> Result<Self> {
        let m = coefficients.nrows();
        if mean.len() != m || covariance.dim() != (m, m) || coefficients.ncols() != intercept.len()
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m}-feature mean/covariance and matching outputs"),
                found: format!(
                    "mean {}, covariance {:?}, intercept {}",
                    mean.len(),
                    covariance.dim(),
                    intercept.len()
                ),
            });
        }
        if covariance
            .iter()
            .chain(mean.iter())
            .chain(coefficients.iter())
            .chain(intercept.iter())
            .any(|v| !v.is_finite())
        {
            return Err(ShapError::InvalidConfiguration(
                "linear parameters must be finite".into(),
            ));
        }
        for i in 0..m {
            for j in 0..m {
                if (covariance[[i, j]] - covariance[[j, i]]).abs() > 1e-10 {
                    return Err(ShapError::InvalidConfiguration(
                        "covariance must be symmetric".into(),
                    ));
                }
            }
        }
        validate_positive_semidefinite(&covariance)?;
        Ok(Self {
            coefficients,
            intercept,
            mean,
            covariance,
            max_features: 16,
            ridge: 1e-10,
        })
    }
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = n;
        self
    }
    pub fn with_ridge(mut self, x: f64) -> Self {
        self.ridge = x;
        self
    }
    fn value(&self, x: ndarray::ArrayView1<'_, f64>, mask: u64) -> Result<Vec<f64>> {
        let m = self.mean.len();
        let present = (0..m).filter(|&j| mask & (1 << j) != 0).collect::<Vec<_>>();
        let absent = (0..m).filter(|&j| mask & (1 << j) == 0).collect::<Vec<_>>();
        let mut expected = self.mean.clone();
        for &j in &present {
            expected[j] = x[j]
        }
        if !present.is_empty() {
            let mut a = vec![vec![0.; present.len()]; present.len()];
            let mut delta = vec![0.; present.len()];
            for (i, &r) in present.iter().enumerate() {
                delta[i] = x[r] - self.mean[r];
                for (j, &c) in present.iter().enumerate() {
                    a[i][j] = self.covariance[[r, c]]
                }
                a[i][i] += self.ridge
            }
            let alpha = solve_vector(a, delta)?;
            for &u in &absent {
                expected[u] = self.mean[u]
                    + present
                        .iter()
                        .enumerate()
                        .map(|(i, &s)| self.covariance[[u, s]] * alpha[i])
                        .sum::<f64>();
            }
        }
        Ok((0..self.intercept.len())
            .map(|o| self.intercept[o] + expected.dot(&self.coefficients.column(o)))
            .collect())
    }
}
impl Explainer for CorrelatedLinearExplainer {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.mean.len();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features"),
                found: format!("{}", x.ncols()),
            });
        }
        if m > self.max_features || m >= 63 {
            return Err(ShapError::InvalidConfiguration(format!(
                "correlated linear SHAP supports at most {} features",
                self.max_features
            )));
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(ShapError::InvalidConfiguration(
                "ridge must be finite and non-negative".into(),
            ));
        }
        let o = self.intercept.len();
        crate::error::checked_f64_shape(&[x.nrows(), m, o], "correlated linear explanation")?;
        let mut values = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        let factorial = (0..=m)
            .scan(1., |v, k| {
                if k > 0 {
                    *v *= k as f64
                }
                Some(*v)
            })
            .collect::<Vec<_>>();
        for n in 0..x.nrows() {
            let mut cache = Vec::with_capacity(1 << m);
            for mask in 0..1u64 << m {
                cache.push(self.value(x.row(n), mask)?)
            }
            for k in 0..o {
                bases[[n, k]] = cache[0][k]
            }
            for j in 0..m {
                for mask in (0..1u64 << m).filter(|z| z & (1 << j) == 0) {
                    let s = mask.count_ones() as usize;
                    let w = factorial[s] * factorial[m - s - 1] / factorial[m];
                    for k in 0..o {
                        values[[n, j, k]] +=
                            w * (cache[(mask | (1 << j)) as usize][k] - cache[mask as usize][k])
                    }
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
    }
}
#[allow(clippy::needless_range_loop)]
fn solve_vector(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>> {
    let n = a.len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        if a[p][c].abs() < 1e-14 {
            return Err(ShapError::SolverError(
                "conditional covariance is singular".into(),
            ));
        }
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        for j in c..n {
            a[c][j] /= d
        }
        b[c] /= d;
        for i in 0..n {
            if i == c {
                continue;
            }
            let f = a[i][c];
            for j in c..n {
                a[i][j] -= f * a[c][j]
            }
            b[i] -= f * b[c]
        }
    }
    Ok(b)
}
fn validate_positive_semidefinite(covariance: &Array2<f64>) -> Result<()> {
    let n = covariance.nrows();
    let scale = (0..n)
        .map(|i| covariance[[i, i]].abs())
        .fold(1.0_f64, f64::max);
    let tolerance = scale * 1e-12;
    let mut lower = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let remainder =
                covariance[[i, j]] - (0..j).map(|k| lower[[i, k]] * lower[[j, k]]).sum::<f64>();
            if i == j {
                if remainder < -tolerance {
                    return Err(ShapError::InvalidConfiguration(
                        "covariance must be positive semidefinite".into(),
                    ));
                }
                lower[[i, j]] = remainder.max(0.0).sqrt();
            } else if lower[[j, j]] > tolerance.sqrt() {
                lower[[i, j]] = remainder / lower[[j, j]];
            } else if remainder.abs() > tolerance {
                return Err(ShapError::InvalidConfiguration(
                    "covariance must be positive semidefinite".into(),
                ));
            }
        }
    }
    Ok(())
}
impl LinearExplainer {
    pub fn new(
        coefficients: Array2<f64>,
        intercept: Array1<f64>,
        background: Background,
    ) -> Result<Self> {
        if coefficients.nrows() != background.n_features()
            || coefficients.ncols() != intercept.len()
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "coefficients ({}, outputs), matching intercept",
                    background.n_features()
                ),
                found: format!(
                    "coefficients {:?}, intercept {}",
                    coefficients.dim(),
                    intercept.len()
                ),
            });
        }
        if coefficients
            .iter()
            .chain(intercept.iter())
            .any(|v| !v.is_finite())
        {
            return Err(ShapError::InvalidConfiguration(
                "linear parameters must be finite".into(),
            ));
        }
        Ok(Self {
            coefficients,
            intercept,
            background,
        })
    }
}
impl Explainer for LinearExplainer {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != self.coefficients.nrows() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.coefficients.nrows()),
                found: format!("{}", x.ncols()),
            });
        }
        let mean = self.background.data().mean_axis(ndarray::Axis(0)).unwrap();
        crate::error::checked_f64_shape(
            &[x.nrows(), x.ncols(), self.intercept.len()],
            "linear explanation",
        )?;
        let mut v = Array3::zeros((x.nrows(), x.ncols(), self.intercept.len()));
        let mut b = Array2::zeros((x.nrows(), self.intercept.len()));
        for i in 0..x.nrows() {
            for o in 0..self.intercept.len() {
                b[[i, o]] = self.intercept[o] + mean.dot(&self.coefficients.column(o));
                for j in 0..x.ncols() {
                    v[[i, j, o]] = (x[[i, j]] - mean[j]) * self.coefficients[[j, o]]
                }
            }
        }
        Explanation::new(v, b, x.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn linear_values_are_closed_form() {
        let background = Background::new(array![[0.0, 2.0], [2.0, 4.0]]).unwrap();
        let explainer =
            LinearExplainer::new(array![[2.0], [-1.0]], array![3.0], background).unwrap();
        let explanation = explainer.explain(array![[3.0, 5.0]].view()).unwrap();
        assert_eq!(explanation.base_values()[[0, 0]], 2.0);
        assert_eq!(explanation.values()[[0, 0, 0]], 4.0);
        assert_eq!(explanation.values()[[0, 1, 0]], -2.0);
        assert_eq!(explanation.reconstructed()[[0, 0]], 4.0);
    }
    #[test]
    fn correlated_linear_matches_independent_case_for_diagonal_covariance() {
        let e = CorrelatedLinearExplainer::new(
            array![[2.], [-1.]],
            array![1.],
            array![0., 0.],
            array![[1., 0.], [0., 1.]],
        )
        .unwrap()
        .explain(array![[3., 4.]].view())
        .unwrap();
        assert!((e.values()[[0, 0, 0]] - 6.).abs() < 1e-8);
        assert!((e.values()[[0, 1, 0]] + 4.).abs() < 1e-8);
        assert!((e.reconstructed()[[0, 0]] - 3.).abs() < 1e-8);
    }

    #[test]
    fn correlated_linear_rejects_indefinite_covariance() {
        let result = CorrelatedLinearExplainer::new(
            array![[1.], [1.]],
            array![0.],
            array![0., 0.],
            array![[1., 2.], [2., 1.]],
        );
        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }

    #[test]
    fn correlated_linear_accepts_singular_covariance() {
        let result = CorrelatedLinearExplainer::new(
            array![[1.], [1.]],
            array![0.],
            array![0., 0.],
            array![[1., 1.], [1., 1.]],
        );
        assert!(result.is_ok());
    }

    #[test]
    fn linear_explainers_reject_non_finite_parameters_and_ridge() {
        let background = Background::new(array![[0., 0.]]).unwrap();
        assert!(LinearExplainer::new(array![[1.], [1.]], array![f64::NAN], background).is_err());

        let correlated =
            CorrelatedLinearExplainer::new(array![[1.]], array![0.], array![0.], array![[1.]])
                .unwrap()
                .with_ridge(-1.);
        assert!(correlated.explain(array![[1.]].view()).is_err());
    }
}
