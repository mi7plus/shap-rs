use crate::{Explanation, Result, ShapError};
use ndarray::ArrayView2;
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct AdditivityTolerance {
    pub absolute: f64,
    pub relative: f64,
}
impl Default for AdditivityTolerance {
    fn default() -> Self {
        Self {
            absolute: 1e-6,
            relative: 1e-6,
        }
    }
}
pub fn additivity_error(e: &Explanation, pred: ArrayView2<'_, f64>) -> Result<f64> {
    e.validate()?;
    let r = e.reconstructed();
    if r.dim() != pred.dim() {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{:?}", r.dim()),
            found: format!("{:?}", pred.dim()),
        });
    }
    if pred.iter().any(|value| !value.is_finite()) {
        return Err(ShapError::ModelError(
            "prediction contains a non-finite value".into(),
        ));
    }
    Ok(r.iter()
        .zip(pred)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max))
}
pub fn check_additivity(e: &Explanation, pred: ArrayView2<'_, f64>, tolerance: f64) -> Result<()> {
    check_additivity_with(
        e,
        pred,
        AdditivityTolerance {
            absolute: tolerance,
            relative: 0.0,
        },
    )
}
pub fn check_additivity_with(
    e: &Explanation,
    pred: ArrayView2<'_, f64>,
    t: AdditivityTolerance,
) -> Result<()> {
    if !t.absolute.is_finite() || !t.relative.is_finite() || t.absolute < 0. || t.relative < 0. {
        return Err(ShapError::InvalidConfiguration(
            "additivity tolerances must be finite and non-negative".into(),
        ));
    }
    e.validate()?;
    let r = e.reconstructed();
    if r.dim() != pred.dim() {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{:?}", r.dim()),
            found: format!("{:?}", pred.dim()),
        });
    }
    if pred.iter().any(|value| !value.is_finite()) {
        return Err(ShapError::ModelError(
            "prediction contains a non-finite value".into(),
        ));
    }
    let mut worst = None;
    for ((i, o), &expected) in pred.indexed_iter() {
        let actual = r[[i, o]];
        let difference = (actual - expected).abs();
        let allowed = t.absolute + t.relative * expected.abs();
        let replace = match worst {
            None => true,
            Some((d, _, _, _)) => difference > d,
        };
        if difference > allowed && replace {
            worst = Some((difference, expected, actual, allowed))
        }
    }
    if let Some((difference, expected, actual, tolerance)) = worst {
        return Err(ShapError::AdditivityError {
            expected,
            actual,
            difference,
            tolerance,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    fn explanation() -> Explanation {
        Explanation::new(
            Array3::from_shape_vec((1, 1, 1), vec![2.]).unwrap(),
            array![[1.]],
            array![[4.]],
        )
        .unwrap()
    }

    #[test]
    fn rejects_non_finite_predictions() {
        let e = explanation();
        assert!(matches!(
            check_additivity(&e, array![[f64::NAN]].view(), 1e-6),
            Err(ShapError::ModelError(_))
        ));
        assert!(additivity_error(&e, array![[f64::INFINITY]].view()).is_err());
    }

    #[test]
    fn supports_combined_absolute_and_relative_tolerance() {
        let e = explanation();
        check_additivity_with(
            &e,
            array![[3.002]].view(),
            AdditivityTolerance {
                absolute: 0.,
                relative: 0.001,
            },
        )
        .unwrap();
        assert!(check_additivity_with(
            &e,
            array![[3.01]].view(),
            AdditivityTolerance {
                absolute: 0.,
                relative: 0.001,
            },
        )
        .is_err());
    }
}
