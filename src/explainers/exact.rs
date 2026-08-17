use crate::{
    coalition, evaluation::CoalitionEvaluator, Background, EvaluationConfig, Explainer,
    Explanation, IndependentMasker, Link, Masker, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2, Axis};
/// Exact interventional Shapley values. Intended for at most ~20 features.
pub struct ExactExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    max_features: usize,
    evaluation: EvaluationConfig,
    link: Link,
}
impl<M> ExactExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background) -> Self {
        Self::from_masker(model, IndependentMasker::new(background))
    }
}
impl<M, K> ExactExplainer<M, K> {
    pub fn from_masker(model: M, masker: K) -> Self {
        Self {
            model,
            masker,
            max_features: 20,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 1 << 20,
                max_model_rows: None,
            },
            link: Link::Identity,
        }
    }
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = n;
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
}
pub(crate) fn checked_predict<M: Predict>(
    model: &M,
    x: ArrayView2<'_, f64>,
) -> Result<Array2<f64>> {
    let y = model.predict(x)?;
    if y.nrows() != x.nrows() || y.ncols() == 0 {
        return Err(ShapError::DimensionMismatch {
            expected: format!("({}, outputs>0)", x.nrows()),
            found: format!("{:?}", y.dim()),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ShapError::ModelError(
            "prediction contains a non-finite value".into(),
        ));
    }
    Ok(y)
}
pub(crate) fn coalition_value<M: Predict, K: Masker>(
    model: &M,
    masker: &K,
    s: ndarray::ArrayView1<'_, f64>,
    mask: &[bool],
) -> Result<Vec<f64>> {
    let y = checked_predict(model, masker.mask(s, mask)?.view())?;
    Ok(y.mean_axis(Axis(0)).unwrap().to_vec())
}
impl<M: Predict, K: Masker> Explainer for ExactExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != self.masker.n_input_features() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} input features", self.masker.n_input_features()),
                found: format!("{} features", x.ncols()),
            });
        }
        if m > self.max_features || m >= 63 {
            return Err(ShapError::InvalidConfiguration(format!(
                "exact SHAP supports at most {} features",
                self.max_features
            )));
        }
        let base = coalition_value(&self.model, &self.masker, x.row(0), &vec![false; m])?;
        let o = base.len();
        crate::error::checked_f64_shape(&[x.nrows(), m, o], "exact explanation")?;
        let mut values = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        let factorial = (0..=m)
            .scan(1.0, |a, k| {
                if k > 0 {
                    *a *= k as f64
                }
                Some(*a)
            })
            .collect::<Vec<_>>();
        for i in 0..x.nrows() {
            let masks = coalition::all(m).collect::<Vec<_>>();
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let cache = evaluator
                .evaluate(x.row(i), &masks)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| self.link.forward(value))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            for out in 0..o {
                bases[[i, out]] = cache[0][out]
            }
            for j in 0..m {
                for mask in coalition::all(m).filter(|z| z & (1 << j) == 0) {
                    let k = mask.count_ones() as usize;
                    let w = factorial[k] * factorial[m - k - 1] / factorial[m];
                    for out in 0..o {
                        values[[i, j, out]] += w
                            * (cache[(mask | (1 << j)) as usize][out] - cache[mask as usize][out]);
                    }
                }
            }
        }
        Explanation::new(values, bases, self.masker.attribution_data(x)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{metrics::check_additivity, FixedMasker, FnModel, GroupedMasker};
    use ndarray::{array, ArrayView2};

    #[test]
    fn exact_values_satisfy_local_accuracy_for_an_interaction() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| r[0] * r[1] + 2.0 * r[0])
                .insert_axis(Axis(1)))
        });
        let background = Background::new(array![[0.0, 0.0], [1.0, 1.0]]).unwrap();
        let x = array![[2.0, 3.0]];
        let explanation = ExactExplainer::new(model, background)
            .explain(x.view())
            .unwrap();

        check_additivity(&explanation, array![[10.0]].view(), 1e-10).unwrap();
        assert_eq!(explanation.values().dim(), (1, 2, 1));
    }
    #[test]
    fn exact_logit_link_explains_log_odds() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.column(0).to_owned().insert_axis(Axis(1))));
        let explanation = ExactExplainer::new(model, Background::new(array![[0.5]]).unwrap())
            .with_link(Link::Logit)
            .explain(array![[0.8]].view())
            .unwrap();
        assert!(explanation.base_values()[[0, 0]].abs() < 1e-12);
        assert!((explanation.values()[[0, 0, 0]] - 4f64.ln()).abs() < 1e-12);
    }
    #[test]
    fn grouped_features_are_explained_as_single_coalition_players() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let masker = GroupedMasker::new(
            FixedMasker::new(array![0., 0., 0.]).unwrap(),
            vec![vec![0, 2], vec![1]],
        )
        .unwrap();
        let explanation = ExactExplainer::from_masker(model, masker)
            .explain(array![[2., 4., 8.]].view())
            .unwrap();
        assert_eq!(explanation.values().dim(), (1, 2, 1));
        assert_eq!(explanation.values(), &array![[[10.], [4.]]]);
        assert_eq!(explanation.data(), array![[5., 4.]].view());
    }
}
