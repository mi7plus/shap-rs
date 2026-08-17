// src/lib.rs
pub mod kernel;
pub mod types;

// Re-export core items for top-level usage
pub use kernel::{explain_sample, generate_coalitions, solve_wls, CoalitionData};
pub use types::{Explainer, Explanation, ShapError};