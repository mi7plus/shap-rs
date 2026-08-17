use crate::{
    evaluation::CoalitionEvaluator, EvaluationConfig, Explainer, Explanation, Link, Masker,
    Predict, Result, ShapError,
};
use ndarray::{Array2, Array3, ArrayView2};

/// Directed acyclic dependency graph for asymmetric causal Shapley values.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CausalGraph {
    parents: Vec<Vec<usize>>,
}
impl CausalGraph {
    pub fn new(parents: Vec<Vec<usize>>) -> Result<Self> {
        let n = parents.len();
        if n == 0 {
            return Err(ShapError::InvalidConfiguration(
                "causal graph cannot be empty".into(),
            ));
        }
        for (i, p) in parents.iter().enumerate() {
            let mut q = p.clone();
            q.sort();
            q.dedup();
            if q.len() != p.len() || p.iter().any(|&j| j >= n || j == i) {
                return Err(ShapError::InvalidConfiguration(
                    "causal graph has invalid or duplicate parents".into(),
                ));
            }
        }
        let graph = Self { parents };
        if !graph.is_acyclic() {
            return Err(ShapError::InvalidConfiguration(
                "causal graph contains a cycle".into(),
            ));
        }
        Ok(graph)
    }
    pub fn parents(&self) -> &[Vec<usize>] {
        &self.parents
    }
    pub fn n_features(&self) -> usize {
        self.parents.len()
    }
    /// Revalidates parent indices, uniqueness, and acyclicity after deserialization.
    pub fn validate(&self) -> Result<()> {
        Self::new(self.parents.clone()).map(|_| ())
    }
    fn is_acyclic(&self) -> bool {
        let mut used = vec![false; self.n_features()];
        for _ in 0..self.n_features() {
            if let Some(j) = (0..self.n_features())
                .find(|&j| !used[j] && self.parents[j].iter().all(|&p| used[p]))
            {
                used[j] = true
            } else {
                return false;
            }
        }
        true
    }
    fn topological_orders(&self, limit: usize) -> Result<Vec<Vec<usize>>> {
        fn rec(
            g: &CausalGraph,
            used: &mut [bool],
            order: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
            limit: usize,
        ) {
            if out.len() > limit {
                return;
            }
            if order.len() == used.len() {
                out.push(order.clone());
                return;
            }
            for j in 0..used.len() {
                if !used[j] && g.parents[j].iter().all(|&p| used[p]) {
                    used[j] = true;
                    order.push(j);
                    rec(g, used, order, out, limit);
                    order.pop();
                    used[j] = false
                }
            }
        }
        let mut out = Vec::new();
        rec(
            self,
            &mut vec![false; self.n_features()],
            &mut Vec::new(),
            &mut out,
            limit,
        );
        if out.len() > limit {
            return Err(ShapError::InvalidConfiguration(format!(
                "causal graph generates more than {limit} topological orders"
            )));
        }
        Ok(out)
    }
}

/// Asymmetric causal Shapley values averaged over all topological orderings.
pub struct CausalExplainer<M, K> {
    model: M,
    masker: K,
    graph: CausalGraph,
    max_orders: usize,
    evaluation: EvaluationConfig,
    link: Link,
}
impl<M, K> CausalExplainer<M, K> {
    pub fn new(model: M, masker: K, graph: CausalGraph) -> Self {
        Self {
            model,
            masker,
            graph,
            max_orders: 65536,
            evaluation: EvaluationConfig {
                coalition_batch_size: 64,
                cache_capacity: 1 << 20,
                max_model_rows: None,
            },
            link: Link::Identity,
        }
    }
    pub fn with_max_orders(mut self, n: usize) -> Self {
        self.max_orders = n;
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
impl<M: Predict, K: Masker> Explainer for CausalExplainer<M, K> {
    fn explain(&self, x: ArrayView2<'_, f64>) -> Result<Explanation> {
        let m = self.masker.n_features();
        self.graph.validate()?;
        if x.nrows() == 0 {
            return Err(ShapError::EmptyData);
        }
        if x.ncols() != m || self.graph.n_features() != m {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{m} features in data and causal graph"),
                found: format!("data {}, graph {}", x.ncols(), self.graph.n_features()),
            });
        }
        if m >= 63 {
            return Err(ShapError::InvalidConfiguration(
                "causal explanations support at most 62 features".into(),
            ));
        }
        let orders = self.graph.topological_orders(self.max_orders)?;
        let mut probe = CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
        let o = probe.evaluate(x.row(0), &[0])?[0].len();
        let mut values = Array3::zeros((x.nrows(), m, o));
        let mut bases = Array2::zeros((x.nrows(), o));
        for n in 0..x.nrows() {
            let mut masks = vec![0u64];
            let mut steps = Vec::with_capacity(orders.len() * m);
            for order in &orders {
                let mut mask = 0;
                let mut before = 0;
                for &j in order {
                    mask |= 1 << j;
                    masks.push(mask);
                    let after = masks.len() - 1;
                    steps.push((j, before, after));
                    before = after
                }
            }
            let mut evaluator =
                CoalitionEvaluator::new(&self.model, &self.masker, self.evaluation)?;
            let evaluated = evaluator
                .evaluate(x.row(n), &masks)?
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
                        (evaluated[after][k] - evaluated[before][k]) / orders.len() as f64
                }
            }
        }
        Explanation::new(values, bases, x.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedMasker, FnModel};
    use ndarray::{array, Axis};
    #[test]
    fn causal_order_allocates_interaction_asymmetrically() {
        let graph = CausalGraph::new(vec![vec![], vec![0]]).unwrap();
        let model = FnModel::new(|x: ArrayView2<'_, f64>| {
            Ok(x.map_axis(Axis(1), |r| r[0] * r[1]).insert_axis(Axis(1)))
        });
        let e = CausalExplainer::new(model, FixedMasker::new(array![0., 0.]).unwrap(), graph)
            .explain(array![[1., 1.]].view())
            .unwrap();
        assert!(e.values()[[0, 0, 0]].abs() < 1e-12);
        assert!((e.values()[[0, 1, 0]] - 1.).abs() < 1e-12);
    }
    #[test]
    fn rejects_invalid_deserialized_style_graph_before_evaluation() {
        let graph = CausalGraph {
            parents: vec![vec![1], vec![0]],
        };
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.sum_axis(Axis(1)).insert_axis(Axis(1))));
        let result = CausalExplainer::new(model, FixedMasker::new(array![0., 0.]).unwrap(), graph)
            .explain(array![[1., 1.]].view());
        assert!(matches!(result, Err(ShapError::InvalidConfiguration(_))));
    }
    #[test]
    fn causal_logit_link_explains_log_odds() {
        let graph = CausalGraph::new(vec![vec![]]).unwrap();
        let model =
            FnModel::new(|x: ArrayView2<'_, f64>| Ok(x.column(0).to_owned().insert_axis(Axis(1))));
        let explanation =
            CausalExplainer::new(model, FixedMasker::new(array![0.5]).unwrap(), graph)
                .with_link(Link::Logit)
                .explain(array![[0.8]].view())
                .unwrap();
        assert!((explanation.reconstructed()[[0, 0]] - 4f64.ln()).abs() < 1e-12);
    }
}
