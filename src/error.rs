//! Error types used throughout `shap-rs`.

use serde::{Deserialize, Serialize};
use std::fmt;

/// The result type used by `shap-rs`.
pub type Result<T> = std::result::Result<T, ShapError>;

pub(crate) fn checked_f64_shape(dimensions: &[usize], context: &str) -> Result<()> {
    let elements = dimensions
        .iter()
        .try_fold(1usize, |size, dimension| size.checked_mul(*dimension));
    let bytes = elements.and_then(|size| size.checked_mul(std::mem::size_of::<f64>()));
    if !matches!(bytes, Some(size) if size <= isize::MAX as usize) {
        return Err(ShapError::InvalidConfiguration(format!(
            "{context} dimensions overflow the addressable allocation size"
        )));
    }
    Ok(())
}

/// Errors that can occur while constructing or evaluating SHAP explainers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapError {
    /// Two or more arrays have incompatible dimensions.
    DimensionMismatch { expected: String, found: String },

    /// A feature index is outside the valid feature range.
    InvalidFeatureIndex { index: usize, n_features: usize },

    /// A sample index is outside the valid sample range.
    InvalidSampleIndex { index: usize, n_samples: usize },

    /// An output index is outside the valid model-output range.
    InvalidOutputIndex { index: usize, n_outputs: usize },

    /// The supplied data contains no samples.
    EmptyData,

    /// The supplied background dataset contains no samples.
    EmptyBackground,

    /// An invalid configuration was supplied to an explainer or component.
    InvalidConfiguration(String),

    /// A model prediction failed.
    ModelError(String),

    /// A masking operation failed.
    MaskerError(String),

    /// A numerical operation failed.
    NumericalError(String),

    /// A linear-system or weighted least-squares solver failed.
    SolverError(String),

    /// The requested functionality is not supported.
    Unsupported(String),

    /// An explanation failed the SHAP additivity check.
    AdditivityError {
        expected: f64,
        actual: f64,
        difference: f64,
        tolerance: f64,
    },

    /// A required feature name or other metadata item was missing.
    MissingMetadata(String),

    /// An operation was requested with incompatible output dimensions.
    OutputDimensionMismatch { expected: usize, found: usize },

    /// An underlying error that does not have a more specific SHAP error type.
    Other(String),
}

impl fmt::Display for ShapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, found } => {
                write!(f, "dimension mismatch: expected {expected}, found {found}")
            }

            Self::InvalidFeatureIndex { index, n_features } => {
                write!(
                    f,
                    "invalid feature index {index}; dataset contains {n_features} features"
                )
            }

            Self::InvalidSampleIndex { index, n_samples } => {
                write!(
                    f,
                    "invalid sample index {index}; explanation contains {n_samples} samples"
                )
            }

            Self::InvalidOutputIndex { index, n_outputs } => {
                write!(
                    f,
                    "invalid output index {index}; explanation contains {n_outputs} outputs"
                )
            }

            Self::EmptyData => {
                write!(f, "input data is empty")
            }

            Self::EmptyBackground => {
                write!(f, "background dataset is empty")
            }

            Self::InvalidConfiguration(message) => {
                write!(f, "invalid configuration: {message}")
            }

            Self::ModelError(message) => {
                write!(f, "model error: {message}")
            }

            Self::MaskerError(message) => {
                write!(f, "masker error: {message}")
            }

            Self::NumericalError(message) => {
                write!(f, "numerical error: {message}")
            }

            Self::SolverError(message) => {
                write!(f, "solver error: {message}")
            }

            Self::Unsupported(message) => {
                write!(f, "unsupported operation: {message}")
            }

            Self::AdditivityError {
                expected,
                actual,
                difference,
                tolerance,
            } => {
                write!(
                    f,
                    "SHAP additivity check failed: expected model output \
                     {expected:.12}, reconstructed output {actual:.12}, \
                     difference {difference:.12}, tolerance {tolerance:.12}"
                )
            }

            Self::MissingMetadata(message) => {
                write!(f, "missing metadata: {message}")
            }

            Self::OutputDimensionMismatch { expected, found } => {
                write!(
                    f,
                    "output dimension mismatch: expected {expected}, found {found}"
                )
            }

            Self::Other(message) => {
                write!(f, "{message}")
            }
        }
    }
}

impl std::error::Error for ShapError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn displays_dimension_mismatch() {
        let error = ShapError::DimensionMismatch {
            expected: "10 features".to_string(),
            found: "8 features".to_string(),
        };

        assert_eq!(
            error.to_string(),
            "dimension mismatch: expected 10 features, found 8 features"
        );
    }

    #[test]
    fn displays_invalid_feature_index() {
        let error = ShapError::InvalidFeatureIndex {
            index: 10,
            n_features: 5,
        };

        assert_eq!(
            error.to_string(),
            "invalid feature index 10; dataset contains 5 features"
        );
    }

    #[test]
    fn displays_invalid_sample_and_output_indices() {
        assert_eq!(
            ShapError::InvalidSampleIndex {
                index: 3,
                n_samples: 2
            }
            .to_string(),
            "invalid sample index 3; explanation contains 2 samples"
        );
        assert_eq!(
            ShapError::InvalidOutputIndex {
                index: 2,
                n_outputs: 1
            }
            .to_string(),
            "invalid output index 2; explanation contains 1 outputs"
        );
    }

    #[test]
    fn displays_additivity_error() {
        let error = ShapError::AdditivityError {
            expected: 0.8,
            actual: 0.7,
            difference: 0.1,
            tolerance: 1e-6,
        };

        let message = error.to_string();

        assert!(message.contains("SHAP additivity check failed"));
        assert!(message.contains("0.800000000000"));
        assert!(message.contains("0.700000000000"));
    }

    #[test]
    fn result_alias_works() {
        fn returns_result() -> Result<()> {
            Ok(())
        }

        assert!(returns_result().is_ok());
    }

    #[test]
    fn rejects_overflowing_or_unaddressable_shapes() {
        assert!(checked_f64_shape(&[usize::MAX, 2], "test").is_err());
        assert!(checked_f64_shape(&[isize::MAX as usize / 8 + 1], "test").is_err());
        assert!(checked_f64_shape(&[2, 3, 4], "test").is_ok());
    }
}
