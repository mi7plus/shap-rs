pub mod acceleration;
pub mod additivity;
pub mod consistency;
pub use acceleration::{
    check_device_equivalence, DeviceEquivalenceReport, DeviceEquivalenceTolerance,
};
pub use additivity::{
    additivity_error, check_additivity, check_additivity_with, AdditivityTolerance,
};
pub use consistency::mean_absolute_shap;
