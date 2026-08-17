use super::{
    model::{MissingBranch, Node, Tree},
    path::{extend, unwind, unwound_sum, PathElement},
};
use ndarray::ArrayView1;
pub(crate) fn tree_shap(tree: &Tree, x: ArrayView1<'_, f64>) -> Vec<Vec<f64>> {
    conditioned_tree_shap(tree, x, usize::MAX, 0)
}
pub(crate) fn conditioned_tree_shap(
    tree: &Tree,
    x: ArrayView1<'_, f64>,
    condition_feature: usize,
    condition: i8,
) -> Vec<Vec<f64>> {
    let mut phi = vec![vec![0.0; tree.n_outputs()]; tree.n_features()];
    recurse(
        tree,
        tree.root(),
        x,
        &mut phi,
        Vec::new(),
        usize::MAX,
        1.0,
        1.0,
        condition_feature,
        condition,
        1.0,
    );
    phi
}
#[allow(clippy::too_many_arguments)]
fn recurse(
    tree: &Tree,
    node: usize,
    x: ArrayView1<'_, f64>,
    phi: &mut [Vec<f64>],
    mut path: Vec<PathElement>,
    parent_feature: usize,
    parent_zero: f64,
    parent_one: f64,
    condition_feature: usize,
    condition: i8,
    condition_fraction: f64,
) {
    if condition_fraction == 0.0 {
        return;
    }
    if condition == 0 || parent_feature != condition_feature {
        extend(&mut path, parent_feature, parent_zero, parent_one);
    }
    match &tree.nodes()[node] {
        Node::Leaf { values, .. } => {
            for i in 1..path.len() {
                let w = unwound_sum(&path, i);
                let e = path[i];
                for (o, v) in values.iter().enumerate() {
                    phi[e.feature][o] += w * (e.one - e.zero) * v * condition_fraction;
                }
            }
        }
        Node::Split {
            feature,
            threshold,
            left,
            right,
            missing,
            ..
        } => {
            let hot = if x[*feature].is_nan() {
                match missing {
                    MissingBranch::Left => *left,
                    MissingBranch::Right => *right,
                }
            } else if x[*feature] <= *threshold {
                *left
            } else {
                *right
            };
            let cold = if hot == *left { *right } else { *left };
            let total = tree.nodes()[*left].cover() + tree.nodes()[*right].cover();
            let hot_zero = if total > 0.0 {
                tree.nodes()[hot].cover() / total
            } else {
                0.5
            };
            let cold_zero = if total > 0.0 {
                tree.nodes()[cold].cover() / total
            } else {
                0.5
            };
            let mut incoming_zero = 1.0;
            let mut incoming_one = 1.0;
            if let Some(i) = path.iter().position(|e| e.feature == *feature) {
                incoming_zero = path[i].zero;
                incoming_one = path[i].one;
                unwind(&mut path, i);
            }
            let mut hot_condition_fraction = condition_fraction;
            let mut cold_condition_fraction = condition_fraction;
            if *feature == condition_feature {
                if condition > 0 {
                    cold_condition_fraction = 0.0;
                } else if condition < 0 {
                    hot_condition_fraction *= hot_zero;
                    cold_condition_fraction *= cold_zero;
                }
            }
            recurse(
                tree,
                hot,
                x,
                phi,
                path.clone(),
                *feature,
                hot_zero * incoming_zero,
                incoming_one,
                condition_feature,
                condition,
                hot_condition_fraction,
            );
            recurse(
                tree,
                cold,
                x,
                phi,
                path,
                *feature,
                cold_zero * incoming_zero,
                0.0,
                condition_feature,
                condition,
                cold_condition_fraction,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coalition;
    use ndarray::Array1;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    fn random_tree(rng: &mut StdRng, depth: usize, m: usize) -> Tree {
        fn build(nodes: &mut Vec<Node>, rng: &mut StdRng, depth: usize, m: usize) -> (usize, f64) {
            let index = nodes.len();
            nodes.push(Node::Leaf {
                values: vec![0., 0.],
                cover: 0.,
            });
            if depth == 0 {
                let cover = rng.gen_range(0.1..5.0);
                nodes[index] = Node::Leaf {
                    values: vec![rng.gen_range(-3.0..3.0), rng.gen_range(-3.0..3.0)],
                    cover,
                };
                return (index, cover);
            }
            let (left, lc) = build(nodes, rng, depth - 1, m);
            let (right, rc) = build(nodes, rng, depth - 1, m);
            nodes[index] = Node::Split {
                feature: rng.gen_range(0..m),
                threshold: rng.gen_range(-1.0..1.0),
                left,
                right,
                missing: if rng.gen_bool(0.5) {
                    MissingBranch::Left
                } else {
                    MissingBranch::Right
                },
                cover: lc + rc,
            };
            (index, lc + rc)
        }
        let mut nodes = Vec::new();
        build(&mut nodes, rng, depth, m);
        Tree::new(nodes, 0, m).unwrap()
    }
    fn brute(tree: &Tree, x: &Array1<f64>) -> Vec<Vec<f64>> {
        let m = tree.n_features();
        let o = tree.n_outputs();
        let mut cache = vec![vec![0.; o]; 1 << m];
        for mask in coalition::all(m) {
            cache[mask as usize] = tree.conditional_value(x.view(), &coalition::members(mask, m))
        }
        let factorial = (0..=m)
            .scan(1., |v, k| {
                if k > 0 {
                    *v *= k as f64
                }
                Some(*v)
            })
            .collect::<Vec<_>>();
        let mut phi = vec![vec![0.; o]; m];
        for j in 0..m {
            for mask in coalition::all(m).filter(|z| z & (1 << j) == 0) {
                let s = mask.count_ones() as usize;
                let w = factorial[s] * factorial[m - s - 1] / factorial[m];
                for k in 0..o {
                    phi[j][k] +=
                        w * (cache[(mask | (1 << j)) as usize][k] - cache[mask as usize][k])
                }
            }
        }
        phi
    }
    #[test]
    fn randomized_tree_shap_matches_brute_force() {
        let mut rng = StdRng::seed_from_u64(0x5A17);
        for _ in 0..100 {
            let tree = random_tree(&mut rng, 3, 4);
            let mut x = Array1::from((0..4).map(|_| rng.gen_range(-2.0..2.0)).collect::<Vec<_>>());
            if rng.gen_bool(0.2) {
                x[rng.gen_range(0..4)] = f64::NAN
            }
            let fast = tree_shap(&tree, x.view());
            let exact = brute(&tree, &x);
            for j in 0..4 {
                for o in 0..2 {
                    assert!(
                        (fast[j][o] - exact[j][o]).abs() < 1e-9,
                        "feature {j}, output {o}: fast {}, exact {}",
                        fast[j][o],
                        exact[j][o]
                    );
                }
            }
        }
    }
}
