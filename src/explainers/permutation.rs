use crate::{
    evaluation::CoalitionEvaluator, Background, EvaluationConfig, Explainer, Explanation,
    IndependentMasker, Link, Masker, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
/// Monte-Carlo Shapley estimator using random feature permutations.
pub struct PermutationExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    n_permutations: usize,
    seed: u64,
    antithetic: bool,
    link: Link,
    evaluation: EvaluationConfig,
}
impl<M> PermutationExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background) -> Self {
        Self::from_masker(model, IndependentMasker::new(background))
    }
}
impl<M, K> PermutationExplainer<M, K> {
    pub fn from_masker(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            n_permutations: 128,
            seed: 0,
            antithetic: true,
            link: Link::Identity,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 65536,
                max_model_rows: None,
            },
        }
    }
    pub fn with_n_permutations(mut self, n: usize) -> Self {
        self.n_permutations = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
        self
    }
    /// Enables reverse-order pairing to reduce Monte Carlo variance.
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
impl<M: Predict, K: Masker> Explainer for PermutationExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features"),
                found: format!("{}", x.ncols()),
            });
        }
        if self.n_permutations == 0 {
            return Err(ShapError::InvalidConfiguration(
                "n_permutations must be positive".into(),
            ));
        }
        if m >= 63 {
            return Err(ShapError::InvalidConfiguration(
                "permutation SHAP currently supports at most 62 features".into(),
            ));
        }
        let mut probe_evaluator =
            CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let o = probe_evaluator.evaluate(x.row(0), &[0])?[0].len();
        let mut vals = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        for i in 0..x.nrows() {
            let mut rng = StdRng::seed_from_u64(crate::coalition::sample_seed(self.seed, x.row(i)));
            let mut requested = vec![0u64];
            let mut steps = Vec::with_capacity(self.n_permutations * m);
            let mut generated = 0;
            while generated < self.n_permutations {
                let mut order = (0..m).collect::<Vec<_>>();
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
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let evaluated = evaluator
                .evaluate(x.row(i), &requested)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| self.link.forward(value))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            let base = &evaluated[0];
            for out in 0..o {
                bases[[i, out]] = base[out]
            }
            for (j, before, after) in steps {
                for out in 0..o {
                    vals[[i, j, out]] += (evaluated[after][out] - evaluated[before][out])
                        / self.n_permutations as f64
                }
            }
        }
        Explanation::new(vals, bases, x.to_owned())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedMasker, FnModel};
    use ndarray::{array, Axis};

    #[test]
    fn antithetic_pair_is_exact_for_two_feature_interaction() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |row| row[0] * row[1])
                .insert_axis(Axis(1)))
        });
        let explanation =
            PermutationExplainer::from_masker(model, FixedMasker::new(array![0., 0.]).unwrap())
                .with_n_permutations(2)
                .with_antithetic(true)
                .explain(array![[2., 3.]].view())
                .unwrap();
        assert!((explanation.values()[[0, 0, 0]] - 3.).abs() < 1e-12);
        assert!((explanation.values()[[0, 1, 0]] - 3.).abs() < 1e-12);
    }

    #[test]
    fn odd_permutation_count_is_preserved() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let explanation =
            PermutationExplainer::from_masker(model, FixedMasker::new(array![0., 0., 0.]).unwrap())
                .with_n_permutations(3)
                .explain(array![[1., 2., 3.]].view())
                .unwrap();
        assert_eq!(explanation.reconstructed(), array![[6.]]);
    }

    #[test]
    fn permutation_logit_link_is_locally_accurate_in_log_odds() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.column(0).to_owned().insert_axis(Axis(1))));
        let explanation =
            PermutationExplainer::from_masker(model, FixedMasker::new(array![0.5]).unwrap())
                .with_n_permutations(2)
                .with_link(Link::Logit)
                .explain(array![[0.8]].view())
                .unwrap();
        assert!((explanation.reconstructed()[[0, 0]] - 4f64.ln()).abs() < 1e-12);
    }
}
