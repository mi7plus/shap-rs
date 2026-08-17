//! Native decision-tree types and the polynomial-time TreeSHAP algorithm.
mod model;
mod path;
mod treeshap;

pub use model::{MissingBranch, Node, Tree, TreeArrays, TreeEnsemble};
#[cfg(feature = "json-adapters")]
pub mod adapters;
pub(crate) use treeshap::{conditioned_tree_shap, tree_shap};
