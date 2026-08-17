//! Optional Burn autodiff integration.
use crate::{DeepAttribution, DifferentiablePredict, Predict, Result, ShapError};
use burn_core::tensor::{backend::AutodiffBackend, Tensor, TensorData};
use ndarray::{Array2, Array3, ArrayView2};

/// Concrete Burn affine graph adapter (`x.matmul(weights) + bias`). This is the
/// deliberately narrow Deep SHAP integration: affine operations are exact;
/// arbitrary Burn graphs should use [`BurnModel`] with `GradientExplainer`.
pub struct BurnAffineModel<B: AutodiffBackend> {
    coefficients: Array2<f64>,
    intercept: Vec<f64>,
    device: B::Device,
}

impl<B: AutodiffBackend> BurnAffineModel<B> {
    pub fn new(coefficients: Array2<f64>, intercept: Vec<f64>, device: B::Device) -> Result<Self> {
        if coefficients.nrows() == 0
            || coefficients.ncols() == 0
            || coefficients.ncols() != intercept.len()
            || coefficients
                .iter()
                .chain(intercept.iter())
                .any(|value| !value.is_finite())
        {
            return Err(ShapError::InvalidConfiguration(
                "Burn affine models require finite feature-by-output weights and matching bias"
                    .into(),
            ));
        }
        Ok(Self {
            coefficients,
            intercept,
            device,
        })
    }
}

impl<B: AutodiffBackend> Predict for BurnAffineModel<B> {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.coefficients.nrows() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.coefficients.nrows()),
                found: x.ncols().to_string(),
            });
        }
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(
                x.iter().map(|&value| value as f32).collect::<Vec<_>>(),
                [x.nrows(), x.ncols()],
            ),
            &self.device,
        );
        let weights = Tensor::<B, 2>::from_data(
            TensorData::new(
                self.coefficients
                    .iter()
                    .map(|&value| value as f32)
                    .collect::<Vec<_>>(),
                [self.coefficients.nrows(), self.coefficients.ncols()],
            ),
            &self.device,
        );
        let bias = Tensor::<B, 2>::from_data(
            TensorData::new(
                self.intercept
                    .iter()
                    .map(|&value| value as f32)
                    .collect::<Vec<_>>(),
                [1, self.intercept.len()],
            ),
            &self.device,
        );
        let values = (input.matmul(weights) + bias)
            .inner()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| ShapError::ModelError(format!("{error:?}")))?;
        Array2::from_shape_vec(
            (x.nrows(), self.intercept.len()),
            values.into_iter().map(f64::from).collect(),
        )
        .map_err(|error| ShapError::ModelError(error.to_string()))
    }
    fn n_features(&self) -> Option<usize> {
        Some(self.coefficients.nrows())
    }
    fn n_outputs(&self) -> Option<usize> {
        Some(self.intercept.len())
    }
}

impl<B: AutodiffBackend> DeepAttribution for BurnAffineModel<B> {
    fn deep_contributions(
        &self,
        x: ArrayView2<'_, f64>,
        background: ArrayView2<'_, f64>,
    ) -> Result<Array3<f64>> {
        if x.ncols() != self.coefficients.nrows()
            || background.ncols() != self.coefficients.nrows()
            || background.nrows() == 0
        {
            return Err(ShapError::DimensionMismatch {
                expected: format!("non-empty data with {} features", self.coefficients.nrows()),
                found: format!("input {:?}, background {:?}", x.dim(), background.dim()),
            });
        }
        let mean = background
            .mean_axis(ndarray::Axis(0))
            .ok_or(ShapError::EmptyBackground)?;
        Ok(Array3::from_shape_fn(
            (x.nrows(), self.coefficients.nrows(), self.intercept.len()),
            |(sample, feature, output)| {
                (x[[sample, feature]] - mean[feature]) * self.coefficients[[feature, output]]
            },
        ))
    }
}

impl<B: AutodiffBackend> DifferentiablePredict for BurnAffineModel<B> {
    fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>> {
        if x.ncols() != self.coefficients.nrows() {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.coefficients.nrows()),
                found: x.ncols().to_string(),
            });
        }
        Ok(Array3::from_shape_fn(
            (
                x.nrows(),
                self.coefficients.nrows(),
                self.coefficients.ncols(),
            ),
            |(_, feature, output)| self.coefficients[[feature, output]],
        ))
    }
}

/// Adapts a Burn tensor forward function to the `shap-rs` model contracts.
pub struct BurnModel<B: AutodiffBackend, F> {
    forward: F,
    device: B::Device,
    n_features: usize,
    n_outputs: usize,
}
impl<B: AutodiffBackend, F> BurnModel<B, F> {
    pub fn new(forward: F, device: B::Device, n_features: usize, n_outputs: usize) -> Result<Self> {
        if n_features == 0 || n_outputs == 0 {
            return Err(ShapError::InvalidConfiguration(
                "Burn models require positive feature and output counts".into(),
            ));
        }
        Ok(Self {
            forward,
            device,
            n_features,
            n_outputs,
        })
    }
    pub fn device(&self) -> &B::Device {
        &self.device
    }
    fn input(&self, x: ArrayView2<'_, f64>) -> Result<Tensor<B, 2>> {
        if x.ncols() != self.n_features {
            return Err(ShapError::DimensionMismatch {
                expected: format!("{} features", self.n_features),
                found: x.ncols().to_string(),
            });
        }
        Ok(Tensor::from_data(
            TensorData::new(
                x.iter().map(|&v| v as f32).collect::<Vec<_>>(),
                [x.nrows(), x.ncols()],
            ),
            &self.device,
        ))
    }
    fn check(&self, output: &Tensor<B, 2>, rows: usize) -> Result<()> {
        if output.dims() != [rows, self.n_outputs] {
            return Err(ShapError::DimensionMismatch {
                expected: format!("({rows}, {}) Burn output", self.n_outputs),
                found: format!("{:?}", output.dims()),
            });
        }
        Ok(())
    }
}
impl<B, F> Predict for BurnModel<B, F>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        let rows = x.nrows();
        let output = (self.forward)(self.input(x)?);
        self.check(&output, rows)?;
        let values = output
            .inner()
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| ShapError::ModelError(format!("{e:?}")))?;
        Array2::from_shape_vec(
            (rows, self.n_outputs),
            values.into_iter().map(f64::from).collect(),
        )
        .map_err(|e| ShapError::ModelError(e.to_string()))
    }
    fn n_features(&self) -> Option<usize> {
        Some(self.n_features)
    }
    fn n_outputs(&self) -> Option<usize> {
        Some(self.n_outputs)
    }
}
impl<B, F> DifferentiablePredict for BurnModel<B, F>
where
    B: AutodiffBackend,
    F: Fn(Tensor<B, 2>) -> Tensor<B, 2>,
{
    fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>> {
        let rows = x.nrows();
        crate::error::checked_f64_shape(
            &[rows, self.n_features, self.n_outputs],
            "Burn gradient output",
        )?;
        let mut result = Array3::zeros((rows, self.n_features, self.n_outputs));
        for output_index in 0..self.n_outputs {
            let input = self.input(x)?.require_grad();
            let output = (self.forward)(input.clone());
            self.check(&output, rows)?;
            let gradients = output
                .slice([0..rows, output_index..output_index + 1])
                .sum()
                .backward();
            let values = input
                .grad(&gradients)
                .ok_or_else(|| {
                    ShapError::ModelError("Burn did not retain the input gradient".into())
                })?
                .into_data()
                .to_vec::<f32>()
                .map_err(|e| ShapError::ModelError(format!("{e:?}")))?;
            for sample in 0..rows {
                for feature in 0..self.n_features {
                    result[[sample, feature, output_index]] =
                        f64::from(values[sample * self.n_features + feature]);
                }
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        explainers::{DeepExplainer, GradientExplainer},
        Background, Explainer,
    };
    use burn_core::backend::Autodiff;
    use burn_ndarray::NdArray;
    use ndarray::array;
    use std::sync::Mutex;
    type Backend = Autodiff<NdArray<f32>>;
    static AUTODIFF_TEST: Mutex<()> = Mutex::new(());
    #[test]
    fn predicts_and_differentiates_multi_output_model() {
        let _guard = AUTODIFF_TEST.lock().unwrap();
        let model = BurnModel::<Backend, _>::new(
            |input: Tensor<Backend, 2>| {
                let rows = input.dims()[0];
                let a = input.clone().slice([0..rows, 0..1]);
                let b = input.slice([0..rows, 1..2]);
                Tensor::cat(vec![a.clone() * 2.0 + b.clone(), a * b], 1)
            },
            Default::default(),
            2,
            2,
        )
        .unwrap();
        let x = array![[3.0, 4.0], [1.0, 2.0]];
        assert_eq!(
            model.predict(x.view()).unwrap(),
            array![[10.0, 12.0], [4.0, 2.0]]
        );
        let gradients = model.gradients(x.view()).unwrap();
        assert_eq!(
            [
                gradients[[0, 0, 0]],
                gradients[[0, 1, 0]],
                gradients[[0, 0, 1]],
                gradients[[0, 1, 1]]
            ],
            [2.0, 1.0, 4.0, 3.0]
        );
    }

    #[test]
    fn drives_expected_gradients_end_to_end() {
        let _guard = AUTODIFF_TEST.lock().unwrap();
        let model = BurnModel::<Backend, _>::new(
            |input: Tensor<Backend, 2>| {
                let rows = input.dims()[0];
                input.clone().slice([0..rows, 0..1]) * 2.0 - input.slice([0..rows, 1..2])
            },
            Default::default(),
            2,
            1,
        )
        .unwrap();
        let explanation =
            GradientExplainer::new(model, Background::new(array![[0.0, 0.0]]).unwrap())
                .with_nsamples(16)
                .with_batch_size(3)
                .explain(array![[3.0, 4.0]].view())
                .unwrap();
        assert!((explanation.values()[[0, 0, 0]] - 6.0).abs() < 1e-6);
        assert!((explanation.values()[[0, 1, 0]] + 4.0).abs() < 1e-6);
    }

    #[test]
    fn affine_graph_drives_deep_explainer_exactly() {
        let _guard = AUTODIFF_TEST.lock().unwrap();
        let model = BurnAffineModel::<Backend>::new(
            array![[2.0, -1.0], [0.5, 3.0]],
            vec![0.25, -2.0],
            Default::default(),
        )
        .unwrap();
        let x = array![[3.0, 4.0]];
        assert_eq!(model.predict(x.view()).unwrap(), array![[8.25, 7.0]]);
        let explanation = DeepExplainer::new(
            model,
            Background::new(array![[0.0, 0.0], [2.0, 2.0]]).unwrap(),
        )
        .explain(x.view())
        .unwrap();
        assert_eq!(explanation.values(), array![[[4.0, -2.0], [1.5, 9.0]]]);
        assert_eq!(explanation.reconstructed(), array![[8.25, 7.0]]);
    }
}
