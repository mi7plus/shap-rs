// src/tree.rs

/// Abstract interface for tree nodes across different crates (smartcore, perpetual, xgboost)
pub trait TreeNode {
    fn is_leaf(&self) -> bool;
    fn feature_index(&self) -> usize;
    fn split_threshold(&self) -> f64;
    fn left(&self) -> Option<&dyn TreeNode>;
    fn right(&self) -> Option<&dyn TreeNode>;
    fn leaf_value(&self) -> f64;
    fn node_weight(&self) -> f64; // Node sample count / cover
}

/// Abstract interface for ensemble models
pub trait TreeEnsemble {
    fn num_trees(&self) -> usize;
    fn get_tree(&self, index: usize) -> &dyn TreeNode;
}

pub struct TreeExplainer<'a, M: TreeEnsemble> {
    model: &'a M,
    base_value: f64,
}

impl<'a, M: TreeEnsemble> TreeExplainer<'a, M> {
    pub fn new(model: &'a M, base_value: f64) -> Self {
        Self { model, base_value }
    }

    pub fn explain_one(&self, sample: &[f64]) -> Vec<f64> {
        let mut phi = vec![0.0; sample.len()];
        for i in 0..self.model.num_trees() {
            let tree = self.model.get_tree(i);
            // TreeSHAP recurrence logic here
            self.recurse(tree, sample, &mut phi, 1.0, 1.0);
        }
        phi
    }

    fn recurse(&self, _node: &dyn TreeNode, _sample: &[f64], _phi: &mut [f64], _zero_frac: f64, _one_frac: f64) {
        // Implement Lundberg et al. TreeSHAP path unwinding
    }
}