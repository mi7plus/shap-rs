use crate::{Result, ShapError};
use serde::{Deserialize, Serialize};

/// Output-space transform used while enforcing SHAP local accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Link {
    #[default]
    Identity,
    Logit,
}
impl Link {
    pub fn forward(self, value: f64) -> Result<f64> {
        match self {
            Self::Identity => Ok(value),
            Self::Logit if value > 0.0 && value < 1.0 => Ok((value / (1.0 - value)).ln()),
            Self::Logit => Err(ShapError::NumericalError(
                "logit link requires a value strictly between zero and one".into(),
            )),
        }
    }
    pub fn inverse(self, value: f64) -> f64 {
        match self {
            Self::Identity => value,
            Self::Logit => {
                if value >= 0.0 {
                    1.0 / (1.0 + (-value).exp())
                } else {
                    let e = value.exp();
                    e / (1.0 + e)
                }
            }
        }
    }
}
