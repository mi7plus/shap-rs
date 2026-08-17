use crate::{Predict, Result, ShapError};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MissingBranch {
    Left,
    Right,
}

/// A node in a binary regression tree. `cover` is the training weight reaching
/// the node and is used to integrate out features that are not observed.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Node {
    Leaf {
        values: Vec<f64>,
        cover: f64,
    },
    Split {
        feature: usize,
        threshold: f64,
        left: usize,
        right: usize,
        missing: MissingBranch,
        cover: f64,
    },
}
/// Framework-neutral columnar representation used by model adapters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeArrays {
    pub features: Vec<Option<usize>>,
    pub thresholds: Vec<f64>,
    pub left_children: Vec<Option<usize>>,
    pub right_children: Vec<Option<usize>>,
    pub missing: Vec<MissingBranch>,
    pub leaf_values: Vec<Option<Vec<f64>>>,
    pub covers: Vec<f64>,
    pub root: usize,
    pub n_features: usize,
}
impl Node {
    pub fn cover(&self) -> f64 {
        match self {
            Self::Leaf { cover, .. } | Self::Split { cover, .. } => *cover,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Tree {
    nodes: Vec<Node>,
    root: usize,
    n_features: usize,
    n_outputs: usize,
}
impl Tree {
    /// Revalidates a tree after deserialization or adapter conversion.
    pub fn validate(&self) -> Result<()> {
        Self::new(self.nodes.clone(), self.root, self.n_features).map(|_| ())
    }
    pub fn from_arrays(a: TreeArrays) -> Result<Self> {
        let n = a.features.len();
        if [
            a.thresholds.len(),
            a.left_children.len(),
            a.right_children.len(),
            a.missing.len(),
            a.leaf_values.len(),
            a.covers.len(),
        ]
        .iter()
        .any(|&x| x != n)
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{n} entries in every tree array"),
                found: "inconsistent tree array lengths".into(),
            });
        }
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            match (&a.features[i], &a.leaf_values[i]) {
                (None, Some(values)) => nodes.push(Node::Leaf {
                    values: values.clone(),
                    cover: a.covers[i],
                }),
                (Some(feature), None) => nodes.push(Node::Split {
                    feature: *feature,
                    threshold: a.thresholds[i],
                    left: a.left_children[i].ok_or_else(|| {
                        ShapError::InvalidConfiguration(format!("split {i} has no left child"))
                    })?,
                    right: a.right_children[i].ok_or_else(|| {
                        ShapError::InvalidConfiguration(format!("split {i} has no right child"))
                    })?,
                    missing: a.missing[i],
                    cover: a.covers[i],
                }),
                _ => {
                    return Err(ShapError::InvalidConfiguration(format!(
                        "node {i} must be exactly one of leaf or split"
                    )))
                }
            }
        }
        Self::new(nodes, a.root, a.n_features)
    }
    pub fn new(nodes: Vec<Node>, root: usize, n_features: usize) -> Result<Self> {
        if nodes.is_empty() || root >= nodes.len() {
            return Err(ShapError::InvalidConfiguration(
                "tree must have a valid root".into(),
            ));
        }
        if n_features == 0 {
            return Err(ShapError::InvalidConfiguration(
                "tree must have at least one feature".into(),
            ));
        }
        let mut outputs = None;
        for (i, node) in nodes.iter().enumerate() {
            if !node.cover().is_finite() || node.cover() < 0.0 {
                return Err(ShapError::InvalidConfiguration(format!(
                    "node {i} has invalid cover"
                )));
            }
            match node {
                Node::Leaf { values, .. } => {
                    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
                        return Err(ShapError::InvalidConfiguration(format!(
                            "leaf {i} has invalid values"
                        )));
                    }
                    if outputs
                        .replace(values.len())
                        .is_some_and(|n| n != values.len())
                    {
                        return Err(ShapError::InvalidConfiguration(
                            "all leaves must have the same output count".into(),
                        ));
                    }
                }
                Node::Split {
                    feature,
                    left,
                    right,
                    threshold,
                    ..
                } => {
                    if *feature >= n_features
                        || *left >= nodes.len()
                        || *right >= nodes.len()
                        || !threshold.is_finite()
                    {
                        return Err(ShapError::InvalidConfiguration(format!(
                            "split node {i} is invalid"
                        )));
                    }
                }
            }
        }
        let n_outputs =
            outputs.ok_or_else(|| ShapError::InvalidConfiguration("tree has no leaves".into()))?;
        let tree = Self {
            nodes,
            root,
            n_features,
            n_outputs,
        };
        tree.validate_graph()?;
        Ok(tree)
    }
    fn validate_graph(&self) -> Result<()> {
        fn visit(t: &Tree, i: usize, state: &mut [u8]) -> Result<()> {
            if state[i] == 1 {
                return Err(ShapError::InvalidConfiguration(
                    "tree contains a cycle".into(),
                ));
            }
            if state[i] == 2 {
                return Err(ShapError::InvalidConfiguration(
                    "tree node has multiple parents".into(),
                ));
            }
            state[i] = 1;
            if let Node::Split { left, right, .. } = &t.nodes[i] {
                visit(t, *left, state)?;
                visit(t, *right, state)?;
            }
            state[i] = 2;
            Ok(())
        }
        let mut state = vec![0; self.nodes.len()];
        visit(self, self.root, &mut state)?;
        if state.contains(&0) {
            return Err(ShapError::InvalidConfiguration(
                "tree contains nodes unreachable from the root".into(),
            ));
        }
        Ok(())
    }
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }
    pub fn root(&self) -> usize {
        self.root
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }
    pub fn predict_row(&self, x: ArrayView1<'_, f64>) -> Result<&[f64]> {
        if x.len() != self.n_features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                found: format!("{}", x.len()),
            });
        }
        let mut i = self.root;
        loop {
            match &self.nodes[i] {
                Node::Leaf { values, .. } => return Ok(values),
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    missing,
                    ..
                } => {
                    i = if x[*feature].is_nan() {
                        match missing {
                            MissingBranch::Left => *left,
                            MissingBranch::Right => *right,
                        }
                    } else if x[*feature] <= *threshold {
                        *left
                    } else {
                        *right
                    }
                }
            }
        }
    }
    pub fn expected_value(&self) -> Vec<f64> {
        fn rec(t: &Tree, i: usize) -> Vec<f64> {
            match &t.nodes[i] {
                Node::Leaf { values, .. } => values.clone(),
                Node::Split { left, right, .. } => {
                    let a = rec(t, *left);
                    let b = rec(t, *right);
                    let total = t.nodes[*left].cover() + t.nodes[*right].cover();
                    let p = if total > 0.0 {
                        t.nodes[*left].cover() / total
                    } else {
                        0.5
                    };
                    a.into_iter()
                        .zip(b)
                        .map(|(x, y)| p * x + (1.0 - p) * y)
                        .collect()
                }
            }
        }
        rec(self, self.root)
    }
    #[cfg(test)]
    pub(crate) fn conditional_value(&self, x: ArrayView1<'_, f64>, present: &[bool]) -> Vec<f64> {
        fn rec(t: &Tree, i: usize, x: ArrayView1<'_, f64>, p: &[bool]) -> Vec<f64> {
            match &t.nodes[i] {
                Node::Leaf { values, .. } => values.clone(),
                Node::Split {
                    feature,
                    threshold,
                    left,
                    right,
                    missing,
                    ..
                } => {
                    if p[*feature] {
                        let c = if x[*feature].is_nan() {
                            match missing {
                                MissingBranch::Left => *left,
                                MissingBranch::Right => *right,
                            }
                        } else if x[*feature] <= *threshold {
                            *left
                        } else {
                            *right
                        };
                        rec(t, c, x, p)
                    } else {
                        let a = rec(t, *left, x, p);
                        let b = rec(t, *right, x, p);
                        let total = t.nodes[*left].cover() + t.nodes[*right].cover();
                        let q = if total > 0.0 {
                            t.nodes[*left].cover() / total
                        } else {
                            0.5
                        };
                        a.into_iter()
                            .zip(b)
                            .map(|(u, v)| q * u + (1.0 - q) * v)
                            .collect()
                    }
                }
            }
        }
        rec(self, self.root, x, present)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TreeEnsemble {
    trees: Vec<(Tree, f64)>,
    base_values: Array1<f64>,
    n_features: usize,
}
impl TreeEnsemble {
    pub fn new(trees: Vec<(Tree, f64)>, base_values: Vec<f64>) -> Result<Self> {
        if trees.is_empty() {
            return Err(ShapError::InvalidConfiguration(
                "ensemble must contain a tree".into(),
            ));
        }
        let nf = trees[0].0.n_features();
        let no = trees[0].0.n_outputs();
        if base_values.len() != no
            || base_values.iter().any(|v| !v.is_finite())
            || trees
                .iter()
                .any(|(t, w)| t.n_features() != nf || t.n_outputs() != no || !w.is_finite())
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("trees with {nf} features and {no} outputs"),
                found: "inconsistent ensemble".into(),
            });
        }
        Ok(Self {
            trees,
            base_values: Array1::from(base_values),
            n_features: nf,
        })
    }
    pub fn trees(&self) -> &[(Tree, f64)] {
        &self.trees
    }
    pub fn base_offset(&self) -> ndarray::ArrayView1<'_, f64> {
        self.base_values.view()
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    pub fn n_outputs(&self) -> usize {
        self.base_values.len()
    }
    pub fn expected_value(&self) -> Array1<f64> {
        let mut v = self.base_values.clone();
        for (t, w) in &self.trees {
            for (o, x) in t.expected_value().into_iter().enumerate() {
                v[o] += w * x
            }
        }
        v
    }
    /// Revalidates all trees and ensemble dimensions after deserialization.
    pub fn validate(&self) -> Result<()> {
        for (t, w) in &self.trees {
            t.validate()?;
            if !w.is_finite() {
                return Err(ShapError::InvalidConfiguration(
                    "tree weight must be finite".into(),
                ));
            }
        }
        Self::new(self.trees.clone(), self.base_values.to_vec()).map(|_| ())
    }
}

impl Predict for TreeEnsemble {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                found: format!("{}", x.ncols()),
            });
        }
        let mut out = Array2::from_shape_fn((x.nrows(), self.base_values.len()), |(_, o)| {
            self.base_values[o]
        });
        for i in 0..x.nrows() {
            for (t, w) in &self.trees {
                for (o, v) in t.predict_row(x.row(i))?.iter().enumerate() {
                    out[[i, o]] += w * v
                }
            }
        }
        Ok(out)
    }
    fn n_features(&self) -> Option<usize> {
        Some(self.n_features)
    }
    fn n_outputs(&self) -> Option<usize> {
        Some(self.base_values.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unreachable_and_shared_nodes() {
        let leaf = || Node::Leaf {
            values: vec![0.],
            cover: 1.,
        };
        let unreachable = Tree::new(
            vec![
                Node::Split {
                    feature: 0,
                    threshold: 0.,
                    left: 1,
                    right: 2,
                    missing: MissingBranch::Left,
                    cover: 2.,
                },
                leaf(),
                leaf(),
                leaf(),
            ],
            0,
            1,
        );
        assert!(unreachable.is_err());

        let shared = Tree::new(
            vec![
                Node::Split {
                    feature: 0,
                    threshold: 0.,
                    left: 1,
                    right: 1,
                    missing: MissingBranch::Left,
                    cover: 2.,
                },
                leaf(),
            ],
            0,
            1,
        );
        assert!(shared.is_err());
    }

    #[test]
    fn rejects_non_finite_ensemble_base_values() {
        let tree = Tree::new(
            vec![Node::Leaf {
                values: vec![1.],
                cover: 1.,
            }],
            0,
            1,
        )
        .unwrap();
        assert!(TreeEnsemble::new(vec![(tree, 1.)], vec![f64::NAN]).is_err());
    }
}
