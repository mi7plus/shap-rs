use crate::{
    coalition, evaluation::CoalitionEvaluator, Background, EvaluationConfig, Explainer,
    Explanation, IndependentMasker, Link, Masker, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::collections::BTreeSet;

/// Linear solver used by Kernel SHAP's constrained weighted least squares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KernelSolver {
    /// Fast, allocation-light solve of the normal equations.
    #[default]
    NormalEquations,
    /// Householder QR on the weighted design matrix. This uses more memory but
    /// avoids squaring the design matrix's condition number.
    HouseholderQr,
}

/// Kernel SHAP with Shapley-kernel weighted least squares and an exact
/// efficiency constraint. Sampled coalitions are complement-paired.
pub struct KernelExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    nsamples: usize,
    seed: u64,
    exact_threshold: usize,
    ridge: f64,
    solver: KernelSolver,
    evaluation: EvaluationConfig,
    link: Link,
}
impl<M> KernelExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background) -> Self {
        Self::from_masker(model, IndependentMasker::new(background))
    }
}
impl<M, K> KernelExplainer<M, K> {
    pub fn from_masker(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            nsamples: 512,
            seed: 0,
            exact_threshold: 12,
            ridge: 1e-10,
            solver: KernelSolver::NormalEquations,
            evaluation: EvaluationConfig::default(),
            link: Link::Identity,
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
    pub fn with_exact_threshold(mut self, n: usize) -> Self {
        self.exact_threshold = n;
        self
    }
    pub fn with_ridge(mut self, ridge: f64) -> Self {
        self.ridge = ridge;
        self
    }
    pub fn with_solver(mut self, solver: KernelSolver) -> Self {
        self.solver = solver;
        self
    }
    pub fn with_evaluation_config(mut self, config: EvaluationConfig) -> Self {
        self.evaluation = config;
        self
    }
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }
    fn coalitions(&self, m: usize) -> Result<Vec<u64>> {
        if self.nsamples == 0 {
            return Err(ShapError::InvalidConfiguration(
                "nsamples must be positive".into(),
            ));
        }
        if m < 63 && m <= self.exact_threshold {
            let count = usize::try_from((1u64 << m) - 2).map_err(|_| {
                ShapError::InvalidConfiguration(
                    "exact Kernel SHAP coalition count exceeds usize".into(),
                )
            })?;
            crate::error::checked_f64_shape(&[count], "Kernel SHAP coalition set")?;
            return Ok((1..(1u64 << m) - 1).collect());
        }
        let full = if m < 64 {
            u64::MAX >> (64 - m)
        } else {
            u64::MAX
        };
        let target = self.nsamples.min(if m < 63 {
            usize::try_from((1u64 << m) - 2).unwrap_or(usize::MAX)
        } else {
            usize::MAX
        });
        crate::error::checked_f64_shape(&[target], "Kernel SHAP coalition set")?;
        let mut set = BTreeSet::new();
        let mut rng = StdRng::seed_from_u64(self.seed);
        while set.len() + 2 <= target {
            let z = rng.gen::<u64>() & full;
            let complement = full ^ z;
            if z != 0 && z != full && !set.contains(&z) && !set.contains(&complement) {
                set.insert(z);
                set.insert(complement);
            }
        }
        while set.len() < target {
            let z = rng.gen::<u64>() & full;
            if z != 0 && z != full {
                set.insert(z);
            }
        }
        Ok(set.into_iter().collect())
    }
}
impl<M: Predict, K: Masker> Explainer for KernelExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if m >= 63 {
            return Err(ShapError::InvalidConfiguration(
                "Kernel SHAP currently supports at most 62 features".into(),
            ));
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(ShapError::InvalidConfiguration(
                "ridge must be finite and non-negative".into(),
            ));
        }
        if x.ncols() != self.masker.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} input features", self.masker.n_input_features()),
                found: format!("{}", x.ncols()),
            });
        }
        let masks = self.coalitions(m)?;
        let full_mask = (1u64 << m) - 1;
        let mut first_eval = CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let probe = first_eval.evaluate(x.row(0), &[0])?.remove(0);
        let o = probe.len();
        crate::error::checked_f64_shape(&[x.nrows(), m, o], "kernel explanation")?;
        let mut v = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        for n in 0..x.nrows() {
            let mut requested = Vec::with_capacity(masks.len() + 2);
            requested.push(0);
            requested.push(full_mask);
            requested.extend_from_slice(&masks);
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let evaluated = evaluator.evaluate(x.row(n), &requested)?;
            let base = evaluated[0]
                .iter()
                .map(|&z| self.link.forward(z))
                .collect::<Result<Vec<_>>>()?;
            let full = evaluated[1]
                .iter()
                .map(|&z| self.link.forward(z))
                .collect::<Result<Vec<_>>>()?;
            for k in 0..o {
                bases[[n, k]] = base[k]
            }
            if m == 1 {
                for k in 0..o {
                    v[[n, 0, k]] = full[k] - base[k]
                }
                continue;
            }
            let p = m - 1;
            crate::error::checked_f64_shape(&[p, p], "Kernel SHAP linear system")?;
            crate::error::checked_f64_shape(&[p, o], "Kernel SHAP right-hand side")?;
            let qr_rows = masks.len().checked_add(p).ok_or_else(|| {
                ShapError::InvalidConfiguration("Kernel SHAP QR row count overflow".into())
            })?;
            if self.solver == KernelSolver::HouseholderQr {
                crate::error::checked_f64_shape(&[qr_rows, p], "Kernel SHAP QR design")?;
                crate::error::checked_f64_shape(&[qr_rows, o], "Kernel SHAP QR response")?;
            }
            let mut a = vec![vec![0.; p]; p];
            let mut b = vec![vec![0.; o]; p];
            let mut qr_a = Vec::with_capacity(qr_rows);
            let mut qr_b = Vec::with_capacity(qr_rows);
            for (row, &mask) in masks.iter().enumerate() {
                let z = coalition::members(mask, m);
                let y = evaluated[row + 2]
                    .iter()
                    .map(|&value| self.link.forward(value))
                    .collect::<Result<Vec<_>>>()?;
                let w = coalition::kernel_weight(m, mask.count_ones() as usize);
                let use_qr = self.solver == KernelSolver::HouseholderQr;
                let sqrt_weight = w.sqrt();
                let mut design_row = if use_qr { vec![0.0; p] } else { Vec::new() };
                let mut response_row = if use_qr { vec![0.0; o] } else { Vec::new() };
                for i in 0..p {
                    let xi = (z[i] as u8 as f64) - (z[m - 1] as u8 as f64);
                    if use_qr {
                        design_row[i] = sqrt_weight * xi;
                    }
                    for j in 0..p {
                        a[i][j] += w * xi * ((z[j] as u8 as f64) - (z[m - 1] as u8 as f64))
                    }
                    for k in 0..o {
                        let target = y[k] - base[k] - (z[m - 1] as u8 as f64) * (full[k] - base[k]);
                        b[i][k] += w * xi * target;
                        if use_qr {
                            response_row[k] = sqrt_weight * target;
                        }
                    }
                }
                if use_qr {
                    qr_a.push(design_row);
                    qr_b.push(response_row);
                }
            }
            let beta = match self.solver {
                KernelSolver::NormalEquations => {
                    for (i, row) in a.iter_mut().enumerate().take(p) {
                        row[i] += self.ridge
                    }
                    solve(a, b)?
                }
                KernelSolver::HouseholderQr => {
                    if self.ridge > 0.0 {
                        let scale = self.ridge.sqrt();
                        for column in 0..p {
                            let mut row = vec![0.0; p];
                            row[column] = scale;
                            qr_a.push(row);
                            qr_b.push(vec![0.0; o]);
                        }
                    }
                    solve_qr(qr_a, qr_b, p)?
                }
            };
            for k in 0..o {
                let mut sum = 0.;
                for j in 0..p {
                    v[[n, j, k]] = beta[j][k];
                    sum += beta[j][k]
                }
                v[[n, m - 1, k]] = full[k] - base[k] - sum
            }
        }
        Explanation::new(v, bases, self.masker.attribution_data(x)?)
    }
}
#[allow(clippy::needless_range_loop)]
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    let o = b[0].len();
    for c in 0..n {
        let p = (c..n)
            .max_by(|&i, &j| a[i][c].abs().total_cmp(&a[j][c].abs()))
            .unwrap();
        if a[p][c].abs() < 1e-14 {
            return Err(ShapError::SolverError(
                "singular Kernel SHAP design; increase nsamples or ridge".into(),
            ));
        }
        a.swap(c, p);
        b.swap(c, p);
        let d = a[c][c];
        for j in c..n {
            a[c][j] /= d
        }
        for k in 0..o {
            b[c][k] /= d
        }
        for i in 0..n {
            if i == c {
                continue;
            }
            let f = a[i][c];
            for j in c..n {
                a[i][j] -= f * a[c][j]
            }
            for k in 0..o {
                b[i][k] -= f * b[c][k]
            }
        }
    }
    Ok(b)
}

#[allow(clippy::needless_range_loop)]
fn solve_qr(mut a: Vec<Vec<f64>>, mut b: Vec<Vec<f64>>, columns: usize) -> Result<Vec<Vec<f64>>> {
    let rows = a.len();
    if rows < columns || columns == 0 || b.len() != rows {
        return Err(ShapError::SolverError(
            "Kernel SHAP QR design is underdetermined".into(),
        ));
    }
    let outputs = b.first().map_or(0, Vec::len);
    if outputs == 0
        || a.iter().any(|row| row.len() != columns)
        || b.iter().any(|row| row.len() != outputs)
    {
        return Err(ShapError::SolverError(
            "Kernel SHAP QR design is ragged or empty".into(),
        ));
    }
    for column in 0..columns {
        let norm = a[column..]
            .iter()
            .map(|row| row[column])
            .fold(0.0_f64, f64::hypot);
        if !norm.is_finite() || norm == 0.0 {
            return Err(ShapError::SolverError(
                "rank-deficient Kernel SHAP design; increase nsamples or ridge".into(),
            ));
        }
        let alpha = if a[column][column] >= 0.0 {
            -norm
        } else {
            norm
        };
        let mut reflector = a[column..]
            .iter()
            .map(|row| row[column])
            .collect::<Vec<_>>();
        reflector[0] -= alpha;
        let reflector_norm = reflector.iter().copied().fold(0.0_f64, f64::hypot);
        if !reflector_norm.is_finite() || reflector_norm == 0.0 {
            return Err(ShapError::SolverError(
                "failed to construct Kernel SHAP QR reflector".into(),
            ));
        }
        for value in &mut reflector {
            *value /= reflector_norm;
        }
        for target_column in column..columns {
            let projection = (column..rows)
                .map(|row| reflector[row - column] * a[row][target_column])
                .sum::<f64>();
            for row in column..rows {
                a[row][target_column] -= 2.0 * reflector[row - column] * projection;
            }
        }
        for output in 0..outputs {
            let projection = (column..rows)
                .map(|row| reflector[row - column] * b[row][output])
                .sum::<f64>();
            for row in column..rows {
                b[row][output] -= 2.0 * reflector[row - column] * projection;
            }
        }
        a[column][column] = alpha;
        for row in column + 1..rows {
            a[row][column] = 0.0;
        }
    }
    let scale = (0..columns)
        .map(|index| a[index][index].abs())
        .fold(0.0_f64, f64::max);
    let tolerance = f64::EPSILON * rows.max(columns) as f64 * scale.max(1.0);
    let mut solution = vec![vec![0.0; outputs]; columns];
    for row in (0..columns).rev() {
        if a[row][row].abs() <= tolerance {
            return Err(ShapError::SolverError(
                "rank-deficient Kernel SHAP design; increase nsamples or ridge".into(),
            ));
        }
        for output in 0..outputs {
            let remainder = (row + 1..columns)
                .map(|column| a[row][column] * solution[column][output])
                .sum::<f64>();
            solution[row][output] = (b[row][output] - remainder) / a[row][row];
        }
    }
    if solution.iter().flatten().any(|value| !value.is_finite()) {
        return Err(ShapError::SolverError(
            "Kernel SHAP QR solution is non-finite".into(),
        ));
    }
    Ok(solution)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::explainers::ExactExplainer;
    use crate::{metrics::check_additivity, FixedMasker, FnModel};
    use ndarray::{array, Array2, Axis};

    #[test]
    fn kernel_wls_recovers_linear_shap_values() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| 2.0 * r[0] - 3.0 * r[1] + r[2])
                .insert_axis(Axis(1)))
        });
        let background = Background::new(array![[0., 0., 0.], [2., 2., 2.]]).unwrap();
        let x = array![[3., 4., 5.]];
        let explanation = KernelExplainer::new(model, background)
            .explain(x.view())
            .unwrap();
        assert!((explanation.values()[[0, 0, 0]] - 4.0).abs() < 1e-7);
        assert!((explanation.values()[[0, 1, 0]] + 9.0).abs() < 1e-7);
        assert!((explanation.values()[[0, 2, 0]] - 4.0).abs() < 1e-7);
        check_additivity(&explanation, array![[-1.]].view(), 1e-9).unwrap();
    }
    #[test]
    fn accepts_custom_maskers() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let masker = FixedMasker::new(array![0., 0.]).unwrap();
        let e = KernelExplainer::from_masker(model, masker)
            .explain(array![[2., 3.]].view())
            .unwrap();
        assert!((e.values().sum() - 5.).abs() < 1e-9);
    }
    #[test]
    fn logit_link_explains_log_odds() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.column(0).mapv(|z| z).insert_axis(Axis(1))));
        let e = KernelExplainer::from_masker(model, FixedMasker::new(array![0.5]).unwrap())
            .with_link(Link::Logit)
            .explain(array![[0.8]].view())
            .unwrap();
        assert!((e.base_values()[[0, 0]]).abs() < 1e-12);
        assert!((e.values()[[0, 0, 0]] - 4f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn exact_coalitions_match_exact_shap_for_nonlinear_multi_output_model() {
        fn predict(x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            Ok(Array2::from_shape_fn((x.nrows(), 2), |(i, output)| {
                let r = x.row(i);
                match output {
                    0 => r[0] * r[1] + r[2].sin() - 0.5 * r[3].powi(2),
                    _ => (r[0] - r[2]) * (r[1] + r[3]) + r[0].exp(),
                }
            }))
        }

        let background = Background::new(array![
            [0.0, -1.0, 0.5, 2.0],
            [1.0, 0.5, -0.5, -1.0],
            [-2.0, 1.5, 1.0, 0.25]
        ])
        .unwrap();
        let samples = array![[0.25, 2.0, -1.0, 0.75], [1.5, -0.25, 0.3, -2.0]];

        let exact = ExactExplainer::new(FnModel::new(predict), background.clone())
            .explain(samples.view())
            .unwrap();
        let kernel = KernelExplainer::new(FnModel::new(predict), background)
            .with_exact_threshold(4)
            .with_ridge(0.0)
            .explain(samples.view())
            .unwrap();

        for (actual, expected) in kernel.values().iter().zip(exact.values()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
        }
        for (actual, expected) in kernel.base_values().iter().zip(exact.base_values()) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn householder_qr_kernel_matches_exact_multi_output_values() {
        fn predict(x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            Ok(Array2::from_shape_fn((x.nrows(), 2), |(row, output)| {
                let values = x.row(row);
                if output == 0 {
                    values[0] * values[1] + values[2]
                } else {
                    values[0] - values[1] * values[2]
                }
            }))
        }
        let background = Background::new(array![[0., 0., 0.], [1., -1., 2.]]).unwrap();
        let samples = array![[2., 3., -1.]];
        let exact = ExactExplainer::new(FnModel::new(predict), background.clone())
            .explain(samples.view())
            .unwrap();
        let kernel = KernelExplainer::new(FnModel::new(predict), background)
            .with_solver(KernelSolver::HouseholderQr)
            .with_ridge(0.0)
            .explain(samples.view())
            .unwrap();
        for (actual, expected) in kernel.values().iter().zip(exact.values()) {
            assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
        }
    }

    #[test]
    fn householder_qr_solves_overdetermined_multi_output_system() {
        let design = vec![
            vec![1.0, 1.0],
            vec![1.0, 1.0 + 1e-8],
            vec![1.0, 1.0 - 1e-8],
            vec![1.0, -1.0],
        ];
        let expected = [[2.0, -1.0], [-3.0, 4.0]];
        let response = design
            .iter()
            .map(|row| {
                (0..2)
                    .map(|output| row[0] * expected[0][output] + row[1] * expected[1][output])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let solution = solve_qr(design, response, 2).unwrap();
        for row in 0..2 {
            for output in 0..2 {
                assert!((solution[row][output] - expected[row][output]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn householder_qr_rejects_rank_deficient_design() {
        assert!(solve_qr(
            vec![vec![1.0, 1.0], vec![2.0, 2.0]],
            vec![vec![1.0], vec![2.0]],
            2,
        )
        .is_err());
    }
}
