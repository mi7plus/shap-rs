use crate::{
    tree::{tree_shap, TreeEnsemble},
    Explainer, Explanation, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2};

/// Exact polynomial-time TreeSHAP for native [`TreeEnsemble`] models.
pub struct TreeExplainer<'a> {
    model: &'a TreeEnsemble,
}
impl<'a> TreeExplainer<'a> {
    pub fn new(model: &'a TreeEnsemble) -> Self {
        Self { model }
    }
}
impl Explainer for TreeExplainer<'_> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.model.n_features();
        let o = self.model.n_outputs();
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
}
