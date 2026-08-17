use super::PermutationExplainer;
use crate::{
    Background, EvaluationConfig, Explainer, Explanation, IndependentMasker, Link, Masker, Predict,
    Result, ShapError, UncertainExplanation,
};
use ndarray::ArrayView2;
/// Monte-Carlo Sampling SHAP. Samples random feature orderings and averages
/// marginal contributions; the execution engine is shared with permutation
/// SHAP because those are the defining samples of the Shapley distribution.
pub struct SamplingExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    nsamples: usize,
    seed: u64,
    antithetic: bool,
    link: Link,
    evaluation: EvaluationConfig,
}
impl<M> SamplingExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background) -> Self {
        Self::from_masker(model, IndependentMasker::new(background))
    }
}
impl<M, K> SamplingExplainer<M, K> {
    pub fn from_masker(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            nsamples: 256,
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
    pub fn with_nsamples(mut self, n: usize) -> Self {
        self.nsamples = n;
        self
    }
    pub fn with_seed(mut self, s: u64) -> Self {
        self.seed = s;
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
    pub fn with_evaluation_config(mut self, c: EvaluationConfig) -> Self {
        self.evaluation = c;
        self
    }
}
impl<M: Predict, K: Masker> SamplingExplainer<M, K> {
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
        for r in 0..repeats {
            runs.push(
                PermutationExplainer::from_masker(&self.model, &self.masker)
                    .with_n_permutations(self.nsamples)
                    .with_seed(self.seed.wrapping_add(r as u64))
                    .with_antithetic(self.antithetic)
                    .with_link(self.link)
                    .with_evaluation_config(self.evaluation)
                    .explain(x)?,
            )
        }
        let shape = runs[0].values().dim();
        let mut mean = ndarray::Array3::zeros(shape);
        for run in &runs {
            ndarray::Zip::from(&mut mean)
                .and(run.values())
                .for_each(|m, &x| *m += x)
        }
        mean.mapv_inplace(|v| v / repeats as f64);
        let mut variance = ndarray::Array3::<f64>::zeros(shape);
        for run in &runs {
            ndarray::Zip::from(&mut variance)
                .and(run.values())
                .and(&mean)
                .for_each(|v, &x, &m| *v += (x - m) * (x - m));
        }
        let standard_errors = variance.mapv(|v| (v / ((repeats - 1) * repeats) as f64).sqrt());
        let explanation = Explanation::new(mean, runs[0].base_values().to_owned(), x.to_owned())?;
        UncertainExplanation::new(explanation, standard_errors, repeats)
    }
}
impl<M: Predict, K: Masker> Explainer for SamplingExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        PermutationExplainer::from_masker(&self.model, &self.masker)
            .with_n_permutations(self.nsamples)
            .with_seed(self.seed)
            .with_antithetic(self.antithetic)
            .with_link(self.link)
            .with_evaluation_config(self.evaluation)
            .explain(x)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedMasker, FnModel};
    use ndarray::{array, Axis};
    #[test]
    fn reports_zero_error_for_order_independent_model() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let e = SamplingExplainer::from_masker(model, FixedMasker::new(array![0., 0.]).unwrap())
            .with_nsamples(8)
            .explain_with_uncertainty(array![[2., 3.]].view(), 4)
            .unwrap();
        assert!(e.standard_errors().iter().all(|x| *x < 1e-12));
        assert!((e.explanation().reconstructed()[[0, 0]] - 5.).abs() < 1e-12);
    }
}
