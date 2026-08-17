use super::{ExactExplainer, KernelExplainer};
use crate::{
    Background, EvaluationConfig, Explainer, Explanation, IndependentMasker, Link, Masker, Predict,
    Result,
};
use ndarray::ArrayView2;

/// Algorithm selected by [`AutoExplainer`] for the current feature count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum AutoAlgorithm {
    Exact,
    Kernel,
}

/// Automatic model-agnostic explainer selection with one shared configuration.
///
/// Exact SHAP is selected when the masker feature count is at most
/// `exact_threshold`; otherwise Kernel SHAP is used. Native tree and linear
/// models should still use their specialized explainers directly.
pub struct AutoExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    exact_threshold: usize,
    kernel_samples: usize,
    seed: u64,
    ridge: f64,
    link: Link,
    evaluation: EvaluationConfig,
}

impl<M> AutoExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background) -> Self {
        Self::from_masker(model, IndependentMasker::new(background))
    }
}

impl<M, K> AutoExplainer<M, K> {
    pub fn from_masker(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            exact_threshold: 12,
            kernel_samples: 512,
            seed: 0,
            ridge: 1e-10,
            link: Link::Identity,
            evaluation: EvaluationConfig::default(),
        }
    }
    pub fn with_exact_threshold(mut self, features: usize) -> Self {
        self.exact_threshold = features;
        self
    }
    pub fn with_kernel_samples(mut self, samples: usize) -> Self {
        self.kernel_samples = samples;
        self
    }
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
    pub fn with_ridge(mut self, ridge: f64) -> Self {
        self.ridge = ridge;
        self
    }
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }
    pub fn with_evaluation_config(mut self, evaluation: EvaluationConfig) -> Self {
        self.evaluation = evaluation;
        self
    }
}

impl<M, K: Masker> AutoExplainer<M, K> {
    pub fn selected_algorithm(&self) -> AutoAlgorithm {
        if self.link == Link::Identity && self.masker.n_features() <= self.exact_threshold {
            AutoAlgorithm::Exact
        } else {
            AutoAlgorithm::Kernel
        }
    }
}

impl<M: Predict, K: Masker> Explainer for AutoExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        match self.selected_algorithm() {
            AutoAlgorithm::Exact => ExactExplainer::from_masker(&self.model, &self.masker)
                .with_max_features(self.exact_threshold)
                .with_link(self.link)
                .with_evaluation_config(self.evaluation)
                .explain(x),
            AutoAlgorithm::Kernel => KernelExplainer::from_masker(&self.model, &self.masker)
                .with_nsamples(self.kernel_samples)
                .with_seed(self.seed)
                .with_exact_threshold(self.exact_threshold)
                .with_ridge(self.ridge)
                .with_link(self.link)
                .with_evaluation_config(self.evaluation)
                .explain(x),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{metrics::check_additivity, FixedMasker, FnModel};
    use ndarray::{array, Axis};

    #[test]
    fn selects_exact_below_threshold() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |row| row[0] * row[1] + row[2])
                .insert_axis(Axis(1)))
        });
        let explainer =
            AutoExplainer::from_masker(model, FixedMasker::new(array![0., 0., 0.]).unwrap());
        assert_eq!(explainer.selected_algorithm(), AutoAlgorithm::Exact);
        let explanation = explainer.explain(array![[2., 3., 4.]].view()).unwrap();
        check_additivity(&explanation, array![[10.]].view(), 1e-12).unwrap();
    }

    #[test]
    fn selects_reproducible_kernel_above_threshold() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |row| row[0] * row[1] + row[2] * row[3])
                .insert_axis(Axis(1)))
        });
        let explainer =
            AutoExplainer::from_masker(model, FixedMasker::new(array![0., 0., 0., 0.]).unwrap())
                .with_exact_threshold(2)
                .with_kernel_samples(8)
                .with_seed(7);
        assert_eq!(explainer.selected_algorithm(), AutoAlgorithm::Kernel);
        let sample = array![[1., 2., 3., 4.]];
        let first = explainer.explain(sample.view()).unwrap();
        let second = explainer.explain(sample.view()).unwrap();
        assert_eq!(first, second);
        check_additivity(&first, array![[14.]].view(), 1e-9).unwrap();
    }
}
