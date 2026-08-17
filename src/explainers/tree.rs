use crate::{
    explainers::ExactExplainer,
    tree::{tree_shap, TreeEnsemble},
    AttributionSemantics, Background, Explainer, Explanation, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2, Axis, Slice};

/// Exact polynomial-time TreeSHAP for native [`TreeEnsemble`] models.
pub struct TreeExplainer<'a> {
    model: &'a TreeEnsemble,
}
impl<'a> TreeExplainer<'a> {
    pub fn new(model: &'a TreeEnsemble) -> Self {
        Self { model }
    }
    /// Explains raw outputs with per-sample base margins replacing the model's
    /// fixed base offset. Tree contributions are unchanged; only base values shift.
    pub fn explain_with_base_margin(
        &self,
        x: ArrayView2<'_, f64>,
        base_margin: ArrayView2<'_, f64>,
    ) -> Result<Explanation> {
        if base_margin.dim() != (x.nrows(), self.model.n_outputs()) {
            return Err(ShapError::DimensionMismatch {
                expected: format!("({}, {}) base margins", x.nrows(), self.model.n_outputs()),
                found: format!("{:?}", base_margin.dim()),
            });
        }
        if base_margin.iter().any(|value| !value.is_finite()) {
            return Err(ShapError::InvalidConfiguration(
                "base margins must be finite".into(),
            ));
        }
        let explanation = self.explain(x)?;
        let mut bases = explanation.base_values().to_owned();
        for row in 0..bases.nrows() {
            for output in 0..bases.ncols() {
                bases[[row, output]] +=
                    base_margin[[row, output]] - self.model.base_offset()[output];
            }
        }
        Explanation::new(
            explanation.values().to_owned(),
            bases,
            explanation.data().to_owned(),
        )
        .map(|explanation| explanation.with_semantics(AttributionSemantics::TreePathDependent))
    }
}

/// Exact interventional TreeSHAP using an explicit background distribution.
/// Unlike [`TreeExplainer`], absent features are replaced from background rows
/// rather than integrated using training-path covers.
pub struct InterventionalTreeExplainer<'a> {
    model: &'a TreeEnsemble,
    background: Background,
    max_features: usize,
}

impl<'a> InterventionalTreeExplainer<'a> {
    pub fn new(model: &'a TreeEnsemble, background: Background) -> Self {
        Self {
            model,
            background,
            max_features: 20,
        }
    }

    pub fn with_max_features(mut self, max_features: usize) -> Self {
        self.max_features = max_features;
        self
    }

    /// Explains probabilities exactly under the supplied background. A
    /// one-output ensemble uses sigmoid; multiple outputs use softmax.
    pub fn explain_probability(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        ExactExplainer::new(
            TransformedTreeModel::probability(self.model),
            self.background.clone(),
        )
        .with_max_features(self.max_features)
        .explain(x)
        .map(|explanation| explanation.with_semantics(AttributionSemantics::Interventional))
    }

    /// Explains binary logistic loss for each sample's target label.
    pub fn explain_binary_log_loss(
        &self,
        x: ArrayView2<'_, f64>,
        targets: &[bool],
    ) -> Result<Explanation> {
        if self.model.n_outputs() != 1 {
            return Err(ShapError::Unsupported(
                "binary log-loss explanations require one raw-margin output".into(),
            ));
        }
        if targets.len() != x.nrows() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} binary targets", x.nrows()),
                found: format!("{} targets", targets.len()),
            });
        }
        let mut explanations = Vec::with_capacity(x.nrows());
        for (sample, &target) in targets.iter().enumerate() {
            explanations.push(
                ExactExplainer::new(
                    TransformedTreeModel::binary_log_loss(self.model, target),
                    self.background.clone(),
                )
                .with_max_features(self.max_features)
                .explain(x.slice_axis(Axis(0), Slice::from(sample..sample + 1)))?,
            );
        }
        Explanation::concatenate(&explanations)
            .map(|explanation| explanation.with_semantics(AttributionSemantics::Interventional))
    }
}

#[derive(Clone, Copy)]
enum TreeTransform {
    Probability,
    BinaryLogLoss(bool),
}

struct TransformedTreeModel<'a> {
    model: &'a TreeEnsemble,
    transform: TreeTransform,
}

impl<'a> TransformedTreeModel<'a> {
    fn probability(model: &'a TreeEnsemble) -> Self {
        Self {
            model,
            transform: TreeTransform::Probability,
        }
    }
    fn binary_log_loss(model: &'a TreeEnsemble, target: bool) -> Self {
        Self {
            model,
            transform: TreeTransform::BinaryLogLoss(target),
        }
    }
}

impl Predict for TransformedTreeModel<'_> {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let raw = self.model.predict(x)?;
        match self.transform {
            TreeTransform::Probability if raw.ncols() == 1 => {
                Ok(raw.mapv(|margin| 1.0 / (1.0 + (-margin).exp())))
            }
            TreeTransform::Probability => {
                let mut probability = raw;
                for mut row in probability.rows_mut() {
                    let maximum = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    row.mapv_inplace(|margin| (margin - maximum).exp());
                    let total = row.sum();
                    row.mapv_inplace(|value| value / total);
                }
                Ok(probability)
            }
            TreeTransform::BinaryLogLoss(target) => Ok(raw.mapv(|margin| {
                margin.max(0.0) + (-margin.abs()).exp().ln_1p() - if target { margin } else { 0.0 }
            })),
        }
    }
    fn n_features(&self) -> Option<usize> {
        Some(self.model.n_features())
    }
    fn n_outputs(&self) -> Option<usize> {
        Some(self.model.n_outputs())
    }
}

impl Explainer for InterventionalTreeExplainer<'_> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        ExactExplainer::new(self.model, self.background.clone())
            .with_max_features(self.max_features)
            .explain(x)
            .map(|explanation| explanation.with_semantics(AttributionSemantics::Interventional))
    }
}
impl Explainer for TreeExplainer<'_> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.model.n_features();
        let o = self.model.n_outputs();
        crate::error::checked_f64_shape(&[x.nrows(), m, o], "tree explanation")?;
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features"),
                found: format!("{}", x.ncols()),
            });
        }
        let base = self.model.expected_value();
        let bases = Array2::from_shape_fn((x.nrows(), o), |(_, k)| base[k]);
        let mut values = Array3::zeros((x.nrows(), m, o));
        for i in 0..x.nrows() {
            for (tree, weight) in self.model.trees() {
                let phi = tree_shap(tree, x.row(i));
                for j in 0..m {
                    for k in 0..o {
                        values[[i, j, k]] += weight * phi[j][k]
                    }
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
            .map(|explanation| explanation.with_semantics(AttributionSemantics::TreePathDependent))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{metrics::check_additivity, MissingBranch, Node, Predict, Tree};
    use ndarray::array;

    fn model() -> TreeEnsemble {
        // if x0 <= 0: 1; else if x1 <= 0: 3; else: 7
        let tree = Tree::new(
            vec![
                Node::Split {
                    feature: 0,
                    threshold: 0.0,
                    left: 1,
                    right: 2,
                    missing: MissingBranch::Left,
                    cover: 10.0,
                },
                Node::Leaf {
                    values: vec![1.0],
                    cover: 4.0,
                },
                Node::Split {
                    feature: 1,
                    threshold: 0.0,
                    left: 3,
                    right: 4,
                    missing: MissingBranch::Left,
                    cover: 6.0,
                },
                Node::Leaf {
                    values: vec![3.0],
                    cover: 2.0,
                },
                Node::Leaf {
                    values: vec![7.0],
                    cover: 4.0,
                },
            ],
            0,
            2,
        )
        .unwrap();
        TreeEnsemble::new(vec![(tree, 1.0)], vec![0.5]).unwrap()
    }

    #[test]
    fn tree_shap_is_additive() {
        let model = model();
        let x = array![[1.0, 1.0], [-1.0, 9.0]];
        let explanation = TreeExplainer::new(&model).explain(x.view()).unwrap();
        let prediction = model.predict(x.view()).unwrap();
        check_additivity(&explanation, prediction.view(), 1e-10).unwrap();
        assert!((explanation.base_values()[[0, 0]] - 4.3).abs() < 1e-12);
    }

    #[test]
    fn missing_values_follow_configured_branch() {
        let model = model();
        let prediction = model.predict(array![[f64::NAN, 2.0]].view()).unwrap();
        assert_eq!(prediction[[0, 0]], 1.5);
    }

    #[test]
    fn base_margins_replace_the_fixed_offset() {
        let model = model();
        let x = array![[1.0, 1.0], [-1.0, 9.0]];
        let margins = array![[2.0], [-3.0]];
        let prediction = model
            .predict_with_base_margin(x.view(), margins.view())
            .unwrap();
        let explanation = TreeExplainer::new(&model)
            .explain_with_base_margin(x.view(), margins.view())
            .unwrap();
        check_additivity(&explanation, prediction.view(), 1e-10).unwrap();
        assert!(model
            .predict_with_base_margin(x.view(), array![[1.0, 2.0]].view())
            .is_err());
    }

    #[test]
    fn interventional_tree_values_use_the_supplied_background() {
        let model = model();
        let background = Background::new(array![[-1.0, -1.0], [1.0, 1.0]]).unwrap();
        let x = array![[1.0, -1.0]];
        let interventional = InterventionalTreeExplainer::new(&model, background.clone())
            .explain(x.view())
            .unwrap();
        let exact = ExactExplainer::new(&model, background)
            .explain(x.view())
            .unwrap();
        assert_eq!(interventional.values(), exact.values());
        assert_eq!(interventional.base_values(), exact.base_values());
        check_additivity(
            &interventional,
            model.predict(x.view()).unwrap().view(),
            1e-10,
        )
        .unwrap();
    }

    #[test]
    fn interventional_probability_and_log_loss_are_additive() {
        let model = model();
        let background = Background::new(array![[-1.0, -1.0], [1.0, 1.0]]).unwrap();
        let x = array![[1.0, -1.0], [-1.0, 2.0]];
        let explainer = InterventionalTreeExplainer::new(&model, background);
        let probability = explainer.explain_probability(x.view()).unwrap();
        let expected_probability = TransformedTreeModel::probability(&model)
            .predict(x.view())
            .unwrap();
        check_additivity(&probability, expected_probability.view(), 1e-10).unwrap();

        let loss = explainer
            .explain_binary_log_loss(x.view(), &[true, false])
            .unwrap();
        for sample in 0..x.nrows() {
            let expected = TransformedTreeModel::binary_log_loss(&model, sample == 0)
                .predict(x.slice_axis(Axis(0), Slice::from(sample..sample + 1)))
                .unwrap();
            assert!((loss.reconstructed()[[sample, 0]] - expected[[0, 0]]).abs() < 1e-10);
        }
    }
}
