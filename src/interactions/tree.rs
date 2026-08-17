use crate::{
    explainers::TreeExplainer,
    interactions::InteractionExplanation,
    tree::{conditioned_tree_shap, Node, TreeEnsemble},
    Explainer, Result, ShapError,
};
use ndarray::{Array4, ArrayView2};

/// Tree-specific name retained as an alias to the unified interaction result.
pub type TreeInteractionExplanation = InteractionExplanation;

/// Exact polynomial-time pairwise Shapley interactions for a native tree ensemble.
pub struct TreeInteractionExplainer<'a> {
    model: &'a TreeEnsemble,
    max_features: usize,
}
impl<'a> TreeInteractionExplainer<'a> {
    pub fn new(model: &'a TreeEnsemble) -> Self {
        Self {
            model,
            max_features: 1024,
        }
    }
    pub fn with_max_features(mut self, n: usize) -> Self {
        self.max_features = n;
        self
    }
    pub fn explain(&self, x: ArrayView2<'_, f64>) -> Result<TreeInteractionExplanation> {
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
        if m > self.max_features {
            return Err(ShapError::InvalidConfiguration(format!(
                "tree interaction matrix supports at most {} configured features",
                self.max_features
            )));
        }
        let main = TreeExplainer::new(self.model).explain(x)?;
        let mut values = Array4::zeros((x.nrows(), m, m, o));
        for n in 0..x.nrows() {
            for (tree, weight) in self.model.trees() {
                let mut used = tree
                    .nodes()
                    .iter()
                    .filter_map(|node| match node {
                        Node::Split { feature, .. } => Some(*feature),
                        Node::Leaf { .. } => None,
                    })
                    .collect::<Vec<_>>();
                used.sort_unstable();
                used.dedup();
                for &condition_feature in &used {
                    let present = conditioned_tree_shap(tree, x.row(n), condition_feature, 1);
                    let absent = conditioned_tree_shap(tree, x.row(n), condition_feature, -1);
                    for feature in 0..m {
                        if feature != condition_feature {
                            for output in 0..o {
                                values[[n, condition_feature, feature, output]] += weight
                                    * (present[feature][output] - absent[feature][output])
                                    / 2.0;
                            }
                        }
                    }
                }
            }
            // Averaging the two conditioned passes is theoretically symmetric;
            // enforce exact symmetry to absorb harmless floating-point drift.
            for i in 0..m {
                for j in i + 1..m {
                    for k in 0..o {
                        let symmetric = (values[[n, i, j, k]] + values[[n, j, i, k]]) / 2.0;
                        values[[n, i, j, k]] = symmetric;
                        values[[n, j, i, k]] = symmetric;
                    }
                }
                for k in 0..o {
                    values[[n, i, i, k]] = main.values()[[n, i, k]]
                        - (0..m)
                            .filter(|&j| j != i)
                            .map(|j| values[[n, i, j, k]])
                            .sum::<f64>();
                }
            }
        }
        TreeInteractionExplanation::new(values, main.base_values().to_owned(), x.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{coalition, MissingBranch, Node, Tree};
    use ndarray::{array, Array1, Array2};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    #[test]
    fn interactions_are_symmetric_and_sum_to_prediction_delta() {
        let tree = Tree::new(
            vec![
                Node::Split {
                    feature: 0,
                    threshold: 0.,
                    left: 1,
                    right: 2,
                    missing: MissingBranch::Left,
                    cover: 2.,
                },
                Node::Leaf {
                    values: vec![0.],
                    cover: 1.,
                },
                Node::Split {
                    feature: 1,
                    threshold: 0.,
                    left: 3,
                    right: 4,
                    missing: MissingBranch::Left,
                    cover: 1.,
                },
                Node::Leaf {
                    values: vec![0.],
                    cover: 0.5,
                },
                Node::Leaf {
                    values: vec![4.],
                    cover: 0.5,
                },
            ],
            0,
            2,
        )
        .unwrap();
        let model = TreeEnsemble::new(vec![(tree, 1.)], vec![0.]).unwrap();
        let e = TreeInteractionExplainer::new(&model)
            .explain(array![[1., 1.]].view())
            .unwrap();
        assert!((e.values()[[0, 0, 1, 0]] - e.values()[[0, 1, 0, 0]]).abs() < 1e-12);
        let sum = e.values().iter().sum::<f64>();
        assert!((e.base_values()[[0, 0]] + sum - 4.).abs() < 1e-12);
        assert_eq!(e.data(), array![[1., 1.]]);
        assert_eq!(e.reconstructed(), array![[4.]]);
        assert_eq!((e.n_samples(), e.n_features(), e.n_outputs()), (1, 2, 1));
    }

    fn random_tree(rng: &mut StdRng, depth: usize, features: usize) -> Tree {
        fn build(
            nodes: &mut Vec<Node>,
            rng: &mut StdRng,
            depth: usize,
            features: usize,
        ) -> (usize, f64) {
            let index = nodes.len();
            nodes.push(Node::Leaf {
                values: vec![0., 0.],
                cover: 0.,
            });
            if depth == 0 {
                let cover = rng.gen_range(0.1..4.0);
                nodes[index] = Node::Leaf {
                    values: vec![rng.gen_range(-2.0..2.0), rng.gen_range(-2.0..2.0)],
                    cover,
                };
                return (index, cover);
            }
            let (left, left_cover) = build(nodes, rng, depth - 1, features);
            let (right, right_cover) = build(nodes, rng, depth - 1, features);
            nodes[index] = Node::Split {
                feature: rng.gen_range(0..features),
                threshold: rng.gen_range(-1.0..1.0),
                left,
                right,
                missing: if rng.gen_bool(0.5) {
                    MissingBranch::Left
                } else {
                    MissingBranch::Right
                },
                cover: left_cover + right_cover,
            };
            (index, left_cover + right_cover)
        }
        let mut nodes = Vec::new();
        build(&mut nodes, rng, depth, features);
        Tree::new(nodes, 0, features).unwrap()
    }

    fn brute_interactions(tree: &Tree, x: &Array1<f64>) -> Array4<f64> {
        let features = tree.n_features();
        let outputs = tree.n_outputs();
        let mut cache = vec![vec![0.; outputs]; 1 << features];
        for mask in coalition::all(features) {
            cache[mask as usize] =
                tree.conditional_value(x.view(), &coalition::members(mask, features));
        }
        let factorial = (0..=features)
            .scan(1.0, |value, index| {
                if index > 0 {
                    *value *= index as f64;
                }
                Some(*value)
            })
            .collect::<Vec<_>>();
        let mut values = Array4::zeros((1, features, features, outputs));
        let main = crate::tree::tree_shap(tree, x.view());
        for first in 0..features {
            for second in first + 1..features {
                for mask in coalition::all(features)
                    .filter(|mask| mask & (1 << first) == 0 && mask & (1 << second) == 0)
                {
                    let size = mask.count_ones() as usize;
                    let weight = factorial[size] * factorial[features - size - 2]
                        / (2.0 * factorial[features - 1]);
                    for output in 0..outputs {
                        let difference = cache[(mask | (1 << first) | (1 << second)) as usize]
                            [output]
                            - cache[(mask | (1 << first)) as usize][output]
                            - cache[(mask | (1 << second)) as usize][output]
                            + cache[mask as usize][output];
                        values[[0, first, second, output]] += weight * difference;
                        values[[0, second, first, output]] += weight * difference;
                    }
                }
            }
        }
        for feature in 0..features {
            for output in 0..outputs {
                values[[0, feature, feature, output]] = main[feature][output]
                    - (0..features)
                        .filter(|&other| other != feature)
                        .map(|other| values[[0, feature, other, output]])
                        .sum::<f64>();
            }
        }
        values
    }

    #[test]
    fn polynomial_interactions_match_randomized_brute_force() {
        let mut rng = StdRng::seed_from_u64(0x01A7_EAC7);
        for _ in 0..50 {
            let tree = random_tree(&mut rng, 3, 4);
            let mut sample =
                Array1::from((0..4).map(|_| rng.gen_range(-2.0..2.0)).collect::<Vec<_>>());
            if rng.gen_bool(0.25) {
                sample[rng.gen_range(0..4)] = f64::NAN;
            }
            let expected = brute_interactions(&tree, &sample);
            let ensemble = TreeEnsemble::new(vec![(tree, 1.)], vec![0., 0.]).unwrap();
            let input = Array2::from_shape_vec((1, 4), sample.to_vec()).unwrap();
            let actual = TreeInteractionExplainer::new(&ensemble)
                .explain(input.view())
                .unwrap();
            for (actual, expected) in actual.values().iter().zip(expected.iter()) {
                assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
            }
        }
    }
}
