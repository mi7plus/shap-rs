use ndarray::{array, Array2, Array3, ArrayView2};
use shap_rs::{
    explainers::GradientExplainer, Background, DifferentiablePredict, Explainer, Predict, Result,
};

struct Linear;
impl Predict for Linear {
    fn predict(&self, x: ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        Ok(Array2::from_shape_fn((x.nrows(), 1), |(i, _)| {
            2.0 * x[[i, 0]] - x[[i, 1]]
        }))
    }
}
impl DifferentiablePredict for Linear {
    fn gradients(&self, x: ArrayView2<'_, f64>) -> Result<Array3<f64>> {
        Ok(Array3::from_shape_fn((x.nrows(), 2, 1), |(_, j, _)| {
            if j == 0 {
                2.0
            } else {
                -1.0
            }
        }))
    }
}
fn main() -> Result<()> {
    let explanation = GradientExplainer::new(Linear, Background::new(array![[0.0, 0.0]])?)
        .with_nsamples(64)
        .explain(array![[3.0, 2.0]].view())?;
    println!("{:?}", explanation.values());
    Ok(())
}
