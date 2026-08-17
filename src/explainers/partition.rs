/// A validated, non-overlapping partition of all input features.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FeaturePartition {
    groups: Vec<Vec<usize>>,
    n_features: usize,
}

use crate::{
    coalition, evaluation::CoalitionEvaluator, Background, EvaluationConfig, Explainer,
    Explanation, IndependentMasker, Link, Masker, Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2};

/// Exact Owen values for a partition of the input features. This preserves
/// group boundaries while allocating each group's contribution among members.
pub struct PartitionExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    partition: FeaturePartition,
    max_features: usize,
    evaluation: EvaluationConfig,
    link: Link,
}
impl<M> PartitionExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background, partition: FeaturePartition) -> Self {
        Self::from_masker(model, IndependentMasker::new(background), partition)
    }
}
impl<M, K> PartitionExplainer<M, K> {
    pub fn from_masker(model: M, masker: K, partition: FeaturePartition) -> Self {
        Self {
            model,
            masker,
            partition,
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
    pub fn with_evaluation_config(mut self, c: EvaluationConfig) -> Self {
        self.evaluation = c;
        self
    }
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }
}
impl<M: Predict, K: Masker> Explainer for PartitionExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        self.partition.validate()?;
        if self.partition.n_features() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("partition for {m} features"),
                found: format!("partition for {} features", self.partition.n_features()),
            });
        }
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
                "exact Owen values support at most {} features",
                self.max_features
            )));
        }
        let masks = coalition::all(m).collect::<Vec<_>>();
        let mut first = CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let o = first.evaluate(x.row(0), &[0])?[0].len();
        let mut values = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        let groups = self.partition.groups();
        let ng = groups.len();
        let group_masks = groups
            .iter()
            .map(|g| g.iter().fold(0u64, |z, &j| z | (1u64 << j)))
            .collect::<Vec<_>>();
        let fg = factorials(ng.max(groups.iter().map(Vec::len).max().unwrap_or(0)));
        for n in 0..x.nrows() {
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let cache = evaluator
                .evaluate(x.row(n), &masks)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| self.link.forward(value))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            for k in 0..o {
                bases[[n, k]] = cache[0][k]
            }
            for (g_index, group) in groups.iter().enumerate() {
                let others = (0..ng).filter(|&g| g != g_index).collect::<Vec<_>>();
                for &feature in group {
                    let peers = group
                        .iter()
                        .copied()
                        .filter(|&j| j != feature)
                        .collect::<Vec<_>>();
                    for outer in 0..(1u64 << others.len()) {
                        let selected_groups = outer.count_ones() as usize;
                        let outer_weight =
                            fg[selected_groups] * fg[ng - selected_groups - 1] / fg[ng];
                        let mut base_mask = 0u64;
                        for (pos, &g) in others.iter().enumerate() {
                            if outer & (1 << pos) != 0 {
                                base_mask |= group_masks[g]
                            }
                        }
                        for inner in 0..(1u64 << peers.len()) {
                            let selected_features = inner.count_ones() as usize;
                            let inner_weight = fg[selected_features]
                                * fg[group.len() - selected_features - 1]
                                / fg[group.len()];
                            let mut mask = base_mask;
                            for (pos, &j) in peers.iter().enumerate() {
                                if inner & (1 << pos) != 0 {
                                    mask |= 1 << j
                                }
                            }
                            for k in 0..o {
                                values[[n, feature, k]] += outer_weight
                                    * inner_weight
                                    * (cache[(mask | (1 << feature)) as usize][k]
                                        - cache[mask as usize][k]);
                            }
                        }
                    }
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
    }
}
fn factorials(n: usize) -> Vec<f64> {
    (0..=n)
        .scan(1.0, |v, k| {
            if k > 0 {
                *v *= k as f64
            }
            Some(*v)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PartitionNode {
    Feature(usize),
    Group(Box<PartitionNode>, Box<PartitionNode>),
}
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PartitionTree {
    root: PartitionNode,
    n_features: usize,
}
impl PartitionTree {
    pub fn new(root: PartitionNode, n_features: usize) -> Result<Self> {
        if n_features == 0 {
            return Err(ShapError::InvalidConfiguration(
                "partition tree must contain features".into(),
            ));
        }
        let mut seen = vec![false; n_features];
        fn visit(n: &PartitionNode, seen: &mut [bool]) -> Result<()> {
            match n {
                PartitionNode::Feature(j) => {
                    if *j >= seen.len() || seen[*j] {
                        return Err(ShapError::InvalidConfiguration(
                            "partition tree must contain each feature exactly once".into(),
                        ));
                    }
                    seen[*j] = true
                }
                PartitionNode::Group(a, b) => {
                    visit(a, seen)?;
                    visit(b, seen)?
                }
            }
            Ok(())
        }
        visit(&root, &mut seen)?;
        if seen.iter().any(|x| !*x) {
            return Err(ShapError::InvalidConfiguration(
                "partition tree must contain each feature exactly once".into(),
            ));
        }
        Ok(Self { root, n_features })
    }
    pub fn root(&self) -> &PartitionNode {
        &self.root
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
    /// Revalidates a hierarchy after deserialization.
    pub fn validate(&self) -> Result<()> {
        Self::new(self.root.clone(), self.n_features).map(|_| ())
    }
    fn permutation_count(&self) -> Option<usize> {
        fn rec(node: &PartitionNode) -> Option<usize> {
            match node {
                PartitionNode::Feature(_) => Some(1),
                PartitionNode::Group(left, right) => {
                    rec(left)?.checked_mul(rec(right)?)?.checked_mul(2)
                }
            }
        }
        rec(&self.root)
    }
    fn permutations(&self) -> Vec<Vec<usize>> {
        fn rec(n: &PartitionNode) -> Vec<Vec<usize>> {
            match n {
                PartitionNode::Feature(j) => vec![vec![*j]],
                PartitionNode::Group(a, b) => {
                    let left = rec(a);
                    let right = rec(b);
                    let mut out = Vec::with_capacity(left.len() * right.len() * 2);
                    for l in &left {
                        for r in &right {
                            let mut lr = l.clone();
                            lr.extend(r);
                            out.push(lr);
                            let mut rl = r.clone();
                            rl.extend(l);
                            out.push(rl)
                        }
                    }
                    out
                }
            }
        }
        rec(&self.root)
    }
}

/// Exact hierarchical Owen values for a binary feature partition tree.
pub struct HierarchicalPartitionExplainer<M, K = IndependentMasker> {
    model: M,
    masker: K,
    tree: PartitionTree,
    max_permutations: usize,
    evaluation: EvaluationConfig,
    link: Link,
}
impl<M> HierarchicalPartitionExplainer<M, IndependentMasker> {
    pub fn new(model: M, background: Background, tree: PartitionTree) -> Self {
        Self::from_masker(model, IndependentMasker::new(background), tree)
    }
}
impl<M, K> HierarchicalPartitionExplainer<M, K> {
    pub fn from_masker(model: M, masker: K, tree: PartitionTree) -> Self {
        Self {
            model,
            masker,
            tree,
            max_permutations: 65536,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 1 << 20,
                max_model_rows: None,
            },
            link: Link::Identity,
        }
    }
    pub fn with_max_permutations(mut self, n: usize) -> Self {
        self.max_permutations = n;
        self
    }
    pub fn with_evaluation_config(mut self, c: EvaluationConfig) -> Self {
        self.evaluation = c;
        self
    }
    pub fn with_link(mut self, link: Link) -> Self {
        self.link = link;
        self
    }
}
impl<M: Predict, K: Masker> Explainer for HierarchicalPartitionExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        self.tree.validate()?;
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m || self.tree.n_features() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features in data and hierarchy"),
                found: format!("data {}, hierarchy {}", x.ncols(), self.tree.n_features()),
            });
        }
        if m >= 63 {
            return Err(ShapError::InvalidConfiguration(
                "hierarchical Owen values support at most 62 features".into(),
            ));
        }
        let permutation_count = self.tree.permutation_count().ok_or_else(|| {
            ShapError::InvalidConfiguration("hierarchy permutation count overflowed".into())
        })?;
        if permutation_count > self.max_permutations {
            return Err(ShapError::InvalidConfiguration(format!(
                "hierarchy generates {} permutations, exceeding limit {}",
                permutation_count, self.max_permutations
            )));
        }
        let permutations = self.tree.permutations();
        let mut probe = CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let o = probe.evaluate(x.row(0), &[0])?[0].len();
        let mut values = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        for n in 0..x.nrows() {
            let mut requested = vec![0u64];
            let mut steps = Vec::with_capacity(permutations.len() * m);
            for order in &permutations {
                let mut mask = 0;
                let mut before = 0;
                for &j in order {
                    mask |= 1 << j;
                    requested.push(mask);
                    let after = requested.len() - 1;
                    steps.push((j, before, after));
                    before = after
                }
            }
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let evaluated = evaluator
                .evaluate(x.row(n), &requested)?
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| self.link.forward(value))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            for k in 0..o {
                bases[[n, k]] = evaluated[0][k]
            }
            for (j, before, after) in steps {
                for k in 0..o {
                    values[[n, j, k]] +=
                        (evaluated[after][k] - evaluated[before][k]) / permutations.len() as f64
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
    }
}

/// Builds a binary hierarchy using average-linkage clustering on absolute
/// Pearson-correlation distance (`1 - |r|`).
pub fn correlation_partition(background: &Background) -> Result<PartitionTree> {
    let m = background.n_features();
    let data = background.data();
    let means = data.mean_axis(ndarray::Axis(0)).unwrap();
    let mut clusters = (0..m)
        .map(|j| (vec![j], PartitionNode::Feature(j)))
        .collect::<Vec<_>>();
    let corr = |a: usize, b: usize| {
        let mut xy = 0.;
        let mut xx = 0.;
        let mut yy = 0.;
        for i in 0..data.nrows() {
            let x = data[[i, a]] - means[a];
            let y = data[[i, b]] - means[b];
            xy += x * y;
            xx += x * x;
            yy += y * y
        }
        if xx == 0. || yy == 0. {
            0.
        } else {
            (xy / (xx * yy).sqrt()).abs()
        }
    };
    while clusters.len() > 1 {
        let mut best = (0, 1, f64::INFINITY);
        for i in 0..clusters.len() {
            for j in i + 1..clusters.len() {
                let mut distance = 0.0;
                for &a in &clusters[i].0 {
                    for &b in &clusters[j].0 {
                        distance += 1.0 - corr(a, b)
                    }
                }
                let d = distance / (clusters[i].0.len() * clusters[j].0.len()) as f64;
                if d < best.2 {
                    best = (i, j, d)
                }
            }
        }
        let (i, j, _) = best;
        let (right_features, right) = clusters.remove(j);
        let (left_features, left) = clusters.remove(i);
        let mut features = left_features;
        features.extend(right_features);
        clusters.push((
            features,
            PartitionNode::Group(Box::new(left), Box::new(right)),
        ))
    }
    PartitionTree::new(clusters.pop().unwrap().1, m)
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use crate::{metrics::check_additivity, FnModel};
    use ndarray::{array, ArrayView2, Axis};
    #[test]
    fn owen_values_respect_groups_and_local_accuracy() {
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| r[0] * r[1] + r[2])
                .insert_axis(Axis(1)))
        });
        let bg = Background::new(array![[0., 0., 0.]]).unwrap();
        let partition = FeaturePartition::new(vec![vec![0, 1], vec![2]], 3).unwrap();
        let e = PartitionExplainer::new(model, bg, partition)
            .explain(array![[1., 1., 1.]].view())
            .unwrap();
        assert!((e.values()[[0, 0, 0]] - 0.5).abs() < 1e-12);
        assert!((e.values()[[0, 1, 0]] - 0.5).abs() < 1e-12);
        assert!((e.values()[[0, 2, 0]] - 1.).abs() < 1e-12);
        check_additivity(&e, array![[2.]].view(), 1e-12).unwrap();
    }
    #[test]
    fn hierarchical_owen_values_are_locally_accurate() {
        let tree = PartitionTree::new(
            PartitionNode::Group(
                Box::new(PartitionNode::Group(
                    Box::new(PartitionNode::Feature(0)),
                    Box::new(PartitionNode::Feature(1)),
                )),
                Box::new(PartitionNode::Feature(2)),
            ),
            3,
        )
        .unwrap();
        assert_eq!(tree.permutations().len(), 4);
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| r[0] * r[2]).insert_axis(Axis(1)))
        });
        let bg = Background::new(array![[0., 0., 0.]]).unwrap();
        let e = HierarchicalPartitionExplainer::new(model, bg, tree)
            .explain(array![[1., 8., 1.]].view())
            .unwrap();
        assert!((e.values()[[0, 0, 0]] - 0.5).abs() < 1e-12);
        assert!(e.values()[[0, 1, 0]].abs() < 1e-12);
        assert!((e.values()[[0, 2, 0]] - 0.5).abs() < 1e-12);
    }
    #[test]
    fn correlation_clustering_contains_every_feature() {
        let bg = Background::new(array![[0., 0., 2.], [1., 1., 1.], [2., 2., 0.]]).unwrap();
        let tree = correlation_partition(&bg).unwrap();
        assert_eq!(tree.n_features(), 3);
        assert_eq!(tree.permutations().len(), 4);
    }
    #[test]
    fn rejects_invalid_deserialized_style_partitions_before_evaluation() {
        let invalid = FeaturePartition {
            groups: vec![vec![0, 0]],
            n_features: 2,
        };
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let result =
            PartitionExplainer::new(model, Background::new(array![[0., 0.]]).unwrap(), invalid)
                .explain(array![[1., 1.]].view());
        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }
    #[test]
    fn checks_hierarchy_permutation_limit_before_generation() {
        fn hierarchy(features: std::ops::Range<usize>) -> PartitionNode {
            let mut nodes = features.map(PartitionNode::Feature).collect::<Vec<_>>();
            while nodes.len() > 1 {
                let right = nodes.pop().unwrap();
                let left = nodes.pop().unwrap();
                nodes.push(PartitionNode::Group(Box::new(left), Box::new(right)));
            }
            nodes.pop().unwrap()
        }
        let tree = PartitionTree::new(hierarchy(0..18), 18).unwrap();
        assert_eq!(tree.permutation_count(), Some(1 << 17));
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let result = HierarchicalPartitionExplainer::new(
            model,
            Background::new(Array2::zeros((1, 18))).unwrap(),
            tree,
        )
        .with_max_permutations(16)
        .explain(Array2::ones((1, 18)).view());
        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }
    #[test]
    fn binary_hierarchy_matches_flat_owen_values_for_two_groups() {
        fn predict(x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
            Ok(Array2::from_shape_fn((x.nrows(), 2), |(i, output)| {
                let r = x.row(i);
                if output == 0 {
                    r[0] * r[2] + r[1].sin() + r[3]
                } else {
                    (r[0] + r[1]) * (r[2] - r[3])
                }
            }))
        }
        let background = Background::new(array![
            [0., 0., 0., 0.],
            [1., -1., 0.5, 2.],
            [-0.5, 2., 1., -1.]
        ])
        .unwrap();
        let sample = array![[2., 0.25, -1., 3.]];
        let flat = PartitionExplainer::new(
            FnModel::new(predict),
            background.clone(),
            FeaturePartition::new(vec![vec![0, 1], vec![2, 3]], 4).unwrap(),
        )
        .explain(sample.view())
        .unwrap();
        let hierarchy = PartitionTree::new(
            PartitionNode::Group(
                Box::new(PartitionNode::Group(
                    Box::new(PartitionNode::Feature(0)),
                    Box::new(PartitionNode::Feature(1)),
                )),
                Box::new(PartitionNode::Group(
                    Box::new(PartitionNode::Feature(2)),
                    Box::new(PartitionNode::Feature(3)),
                )),
            ),
            4,
        )
        .unwrap();
        let nested =
            HierarchicalPartitionExplainer::new(FnModel::new(predict), background, hierarchy)
                .explain(sample.view())
                .unwrap();
        for (actual, expected) in nested.values().iter().zip(flat.values()) {
            assert!((actual - expected).abs() < 1e-12);
        }
        assert_eq!(nested.base_values(), flat.base_values());
    }
    #[test]
    fn partition_logit_link_explains_log_odds() {
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.column(0).to_owned().insert_axis(Axis(1))));
        let explanation = PartitionExplainer::new(
            model,
            Background::new(array![[0.5]]).unwrap(),
            FeaturePartition::new(vec![vec![0]], 1).unwrap(),
        )
        .with_link(Link::Logit)
        .explain(array![[0.8]].view())
        .unwrap();
        assert!((explanation.reconstructed()[[0, 0]] - 4f64.ln()).abs() < 1e-12);
    }
}
impl FeaturePartition {
    pub fn new(groups: Vec<Vec<usize>>, n_features: usize) -> crate::Result<Self> {
        if n_features == 0 {
            return Err(crate::ShapError::InvalidConfiguration(
                "partition must contain at least one feature".into(),
            ));
        }
        let mut seen = vec![false; n_features];
        for g in &groups {
            if g.is_empty() {
                return Err(crate::ShapError::InvalidConfiguration(
                    "partition groups cannot be empty".into(),
                ));
            }
            for &j in g {
                if j >= n_features || seen[j] {
                    return Err(crate::ShapError::InvalidConfiguration(
                        "partition must contain every feature exactly once".into(),
                    ));
                }
                seen[j] = true
            }
        }
        if seen.iter().any(|x| !*x) {
            return Err(crate::ShapError::InvalidConfiguration(
                "partition must contain every feature exactly once".into(),
            ));
        }
        Ok(Self { groups, n_features })
    }
    /// Revalidates a partition after deserialization.
    pub fn validate(&self) -> crate::Result<()> {
        Self::new(self.groups.clone(), self.n_features).map(|_| ())
    }
    pub fn groups(&self) -> &[Vec<usize>] {
        &self.groups
    }
    pub fn n_features(&self) -> usize {
        self.n_features
    }
}
