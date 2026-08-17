pub mod additivity;
pub mod consistency;
pub use additivity::{
    additivity_error, check_additivity, check_additivity_with, AdditivityTolerance,
};
pub use consistency::mean_absolute_shap;
