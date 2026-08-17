use crate::{Result, ShapError};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FeatureKind {
    #[default]
    Continuous,
    Categorical,
    Ordinal,
    Boolean,
    TextToken,
    ImageValue,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OutputKind {
    #[default]
    Regression,
    Probability,
    LogOdds,
    ClassScore,
    Embedding,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(try_from = "FeatureMetadataPayload")]
pub struct FeatureMetadata {
    pub names: Vec<String>,
    pub display_names: Option<Vec<String>>,
    pub kinds: Option<Vec<FeatureKind>>,
    pub units: Option<Vec<Option<String>>>,
}
#[derive(Deserialize)]
struct FeatureMetadataPayload {
    names: Vec<String>,
    display_names: Option<Vec<String>>,
    kinds: Option<Vec<FeatureKind>>,
    units: Option<Vec<Option<String>>>,
}
impl TryFrom<FeatureMetadataPayload> for FeatureMetadata {
    type Error = ShapError;
    fn try_from(payload: FeatureMetadataPayload) -> Result<Self> {
        let metadata = Self {
            names: payload.names,
            display_names: payload.display_names,
            kinds: payload.kinds,
            units: payload.units,
        };
        metadata.validate()?;
        Ok(metadata)
    }
}
impl FeatureMetadata {
    pub fn new(names: Vec<String>) -> Result<Self> {
        if names.iter().any(|n| n.is_empty()) {
            return Err(ShapError::InvalidConfiguration(
                "feature names cannot be empty".into(),
            ));
        }
        let mut unique = names.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != names.len() {
            return Err(ShapError::InvalidConfiguration(
                "feature names must be unique".into(),
            ));
        }
        let metadata = Self {
            names,
            display_names: None,
            kinds: None,
            units: None,
        };
        metadata.validate()?;
        Ok(metadata)
    }
    pub fn with_kinds(mut self, kinds: Vec<FeatureKind>) -> Result<Self> {
        if kinds.len() != self.names.len() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} feature kinds", self.names.len()),
                found: format!("{}", kinds.len()),
            });
        }
        self.kinds = Some(kinds);
        Ok(self)
    }
    pub fn with_units(mut self, units: Vec<Option<String>>) -> Result<Self> {
        if units.len() != self.names.len() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} feature units", self.names.len()),
                found: format!("{}", units.len()),
            });
        }
        self.units = Some(units);
        Ok(self)
    }
    pub fn with_display_names(mut self, names: Vec<String>) -> Result<Self> {
        if names.len() != self.names.len() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} display names", self.names.len()),
                found: format!("{}", names.len()),
            });
        }
        self.display_names = Some(names);
        Ok(self)
    }

    /// Validates names and the lengths of every optional metadata field.
    pub fn validate(&self) -> Result<()> {
        if self.names.is_empty() || self.names.iter().any(|name| name.is_empty()) {
            return Err(ShapError::InvalidConfiguration(
                "feature names cannot be empty".into(),
            ));
        }
        let mut unique = self.names.clone();
        unique.sort();
        unique.dedup();
        if unique.len() != self.names.len() {
            return Err(ShapError::InvalidConfiguration(
                "feature names must be unique".into(),
            ));
        }
        if self
            .display_names
            .as_ref()
            .is_some_and(|values| values.len() != self.names.len())
            || self
                .kinds
                .as_ref()
                .is_some_and(|values| values.len() != self.names.len())
            || self
                .units
                .as_ref()
                .is_some_and(|values| values.len() != self.names.len())
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!(
                    "{} entries in all feature metadata fields",
                    self.names.len()
                ),
                found: "inconsistent feature metadata".into(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(try_from = "OutputMetadataPayload")]
pub struct OutputMetadata {
    pub names: Vec<String>,
    pub kinds: Option<Vec<OutputKind>>,
}
#[derive(Deserialize)]
struct OutputMetadataPayload {
    names: Vec<String>,
    kinds: Option<Vec<OutputKind>>,
}
impl TryFrom<OutputMetadataPayload> for OutputMetadata {
    type Error = ShapError;
    fn try_from(payload: OutputMetadataPayload) -> Result<Self> {
        let metadata = Self {
            names: payload.names,
            kinds: payload.kinds,
        };
        metadata.validate()?;
        Ok(metadata)
    }
}
impl OutputMetadata {
    pub fn new(names: Vec<String>) -> Result<Self> {
        if names.iter().any(|n| n.is_empty()) {
            return Err(ShapError::InvalidConfiguration(
                "output names cannot be empty".into(),
            ));
        }
        let metadata = Self { names, kinds: None };
        metadata.validate()?;
        Ok(metadata)
    }
    pub fn with_kinds(mut self, kinds: Vec<OutputKind>) -> Result<Self> {
        if kinds.len() != self.names.len() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} output kinds", self.names.len()),
                found: format!("{}", kinds.len()),
            });
        }
        self.kinds = Some(kinds);
        Ok(self)
    }

    /// Validates output names and optional output kinds.
    pub fn validate(&self) -> Result<()> {
        if self.names.is_empty() || self.names.iter().any(|name| name.is_empty()) {
            return Err(ShapError::InvalidConfiguration(
                "output names cannot be empty".into(),
            ));
        }
        if self
            .kinds
            .as_ref()
            .is_some_and(|values| values.len() != self.names.len())
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} output kinds", self.names.len()),
                found: "inconsistent output metadata".into(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_directly_constructed_feature_metadata() {
        let metadata = FeatureMetadata {
            names: vec!["a".into(), "b".into()],
            display_names: Some(vec!["A".into()]),
            kinds: None,
            units: None,
        };
        assert!(matches!(
            metadata.validate(),
            Err(ShapError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn validates_directly_constructed_output_metadata() {
        let metadata = OutputMetadata {
            names: vec!["prediction".into()],
            kinds: Some(vec![OutputKind::Regression, OutputKind::Probability]),
        };
        assert!(matches!(
            metadata.validate(),
            Err(ShapError::DimensionMismatch { .. })
        ));
    }
}
