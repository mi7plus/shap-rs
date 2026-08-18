use crate::{AcceleratedPredict, ExecutionDevice, Result, ShapError};
use ndarray::{Array2, ArrayView2};

/// Numerical and repeatability requirements for an accelerated prediction path.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceEquivalenceTolerance {
    pub absolute: f64,
    pub relative: f64,
    pub determinism_absolute: f64,
    pub determinism_relative: f64,
    /// Total number of accelerated runs, including the reference device run.
    pub repeated_runs: usize,
}

impl DeviceEquivalenceTolerance {
    /// Recommended contract for the selected execution device.
    ///
    /// CPU execution must repeat bit-for-bit. GPU-style devices permit `1e-5`
    /// CPU equivalence and `1e-6` repeated-run drift, accommodating common f32
    /// reduction and operation-order differences without accepting material
    /// attribution changes.
    pub fn for_device(device: ExecutionDevice) -> Self {
        match device {
            ExecutionDevice::Cpu => Self {
                absolute: 0.0,
                relative: 0.0,
                determinism_absolute: 0.0,
                determinism_relative: 0.0,
                repeated_runs: 2,
            },
            ExecutionDevice::Cuda(_)
            | ExecutionDevice::Metal
            | ExecutionDevice::Vulkan
            | ExecutionDevice::WebGpu => Self {
                absolute: 1e-5,
                relative: 1e-5,
                determinism_absolute: 1e-6,
                determinism_relative: 1e-6,
                repeated_runs: 3,
            },
        }
    }

    pub fn validate(self) -> Result<Self> {
        if self.repeated_runs == 0
            || [
                self.absolute,
                self.relative,
                self.determinism_absolute,
                self.determinism_relative,
            ]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(ShapError::InvalidConfiguration(
                "device tolerances must be finite and non-negative and repeated_runs must be positive"
                    .into(),
            ));
        }
        Ok(self)
    }
}

/// Worst observed CPU/device and repeated-device differences.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DeviceEquivalenceReport {
    pub max_cpu_device_difference: f64,
    pub max_repeated_device_difference: f64,
    pub compared_values: usize,
    pub device_runs: usize,
}

/// Validates an accelerated model against its CPU path and repeated device runs.
pub fn check_device_equivalence<M: AcceleratedPredict>(
    model: &M,
    input: ArrayView2<'_, f64>,
    device: ExecutionDevice,
    tolerance: DeviceEquivalenceTolerance,
) -> Result<DeviceEquivalenceReport> {
    let tolerance = tolerance.validate()?;
    let cpu = model.predict(input)?;
    validate_prediction(&cpu, input.nrows(), "CPU")?;
    let first = model.predict_on(input, device)?;
    validate_prediction(&first, input.nrows(), "accelerated")?;
    if first.dim() != cpu.dim() {
        return Err(ShapError::DimensionMismatch {
            expected: format!("{:?} CPU prediction", cpu.dim()),
            found: format!("{:?} accelerated prediction", first.dim()),
        });
    }
    let max_cpu_device_difference = compare(
        cpu.view(),
        first.view(),
        tolerance.absolute,
        tolerance.relative,
        "CPU/device",
    )?;
    let mut max_repeated_device_difference = 0.0_f64;
    for run in 1..tolerance.repeated_runs {
        let repeated = model.predict_on(input, device)?;
        validate_prediction(&repeated, input.nrows(), "repeated accelerated")?;
        if repeated.dim() != first.dim() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{:?} first device prediction", first.dim()),
                found: format!("{:?} device prediction on run {}", repeated.dim(), run + 1),
            });
        }
        max_repeated_device_difference = max_repeated_device_difference.max(compare(
            first.view(),
            repeated.view(),
            tolerance.determinism_absolute,
            tolerance.determinism_relative,
            "repeated device",
        )?);
    }
    Ok(DeviceEquivalenceReport {
        max_cpu_device_difference,
        max_repeated_device_difference,
        compared_values: cpu.len(),
        device_runs: tolerance.repeated_runs,
    })
}

fn validate_prediction(prediction: &Array2<f64>, rows: usize, label: &str) -> Result<()> {
    if prediction.nrows() != rows || prediction.ncols() == 0 {
        return Err(ShapError::DimensionMismatch {
            expected: format!("({rows}, outputs>0) {label} prediction"),
            found: format!("{:?}", prediction.dim()),
        });
    }
    if prediction.iter().any(|value| !value.is_finite()) {
        return Err(ShapError::ModelError(format!(
            "{label} prediction contains a non-finite value"
        )));
    }
    Ok(())
}

fn compare(
    expected: ArrayView2<'_, f64>,
    actual: ArrayView2<'_, f64>,
    absolute: f64,
    relative: f64,
    label: &str,
) -> Result<f64> {
    let mut maximum = 0.0_f64;
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let difference = (actual - expected).abs();
        let allowed = absolute + relative * expected.abs().max(actual.abs());
        maximum = maximum.max(difference);
        if difference > allowed {
            return Err(ShapError::ModelError(format!(
                "{label} prediction differs at flat index {index}: expected {expected}, actual {actual}, difference {difference}, tolerance {allowed}"
            )));
        }
    }
    Ok(maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FnAcceleratedModel;
    use ndarray::array;
    use std::cell::Cell;

    #[test]
    fn accepts_device_results_within_equivalence_contract() {
        let model = FnAcceleratedModel::new(|x: ArrayView2<'_, f64>, device| {
            let offset = if matches!(device, ExecutionDevice::Cuda(_)) {
                5e-6
            } else {
                0.0
            };
            Ok(x.mapv(|value| value + offset))
        });
        let report = check_device_equivalence(
            &model,
            array![[1.0, 2.0]].view(),
            ExecutionDevice::Cuda(0),
            DeviceEquivalenceTolerance::for_device(ExecutionDevice::Cuda(0)),
        )
        .unwrap();
        assert!((report.max_cpu_device_difference - 5e-6).abs() < 1e-12);
        assert_eq!(report.max_repeated_device_difference, 0.0);
        assert_eq!(report.device_runs, 3);
    }

    #[test]
    fn rejects_device_drift_between_repeated_runs() {
        let calls = Cell::new(0usize);
        let model = FnAcceleratedModel::new(|x: ArrayView2<'_, f64>, device| {
            let offset = if matches!(device, ExecutionDevice::Cuda(_)) {
                let next = calls.get() + 1;
                calls.set(next);
                next as f64 * 1e-3
            } else {
                0.0
            };
            Ok(x.mapv(|value| value + offset))
        });
        assert!(check_device_equivalence(
            &model,
            array![[1.0]].view(),
            ExecutionDevice::Cuda(0),
            DeviceEquivalenceTolerance {
                absolute: 1.0,
                relative: 0.0,
                determinism_absolute: 1e-6,
                determinism_relative: 0.0,
                repeated_runs: 2,
            },
        )
        .is_err());
    }

    #[test]
    fn rejects_invalid_device_tolerances() {
        assert!(DeviceEquivalenceTolerance {
            absolute: -1.0,
            relative: 0.0,
            determinism_absolute: 0.0,
            determinism_relative: 0.0,
            repeated_runs: 1,
        }
        .validate()
        .is_err());
    }
}
